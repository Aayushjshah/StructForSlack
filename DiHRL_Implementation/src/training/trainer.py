"""
Training script for DiHRL model.
Implements GPU-compatible training with hierarchical ranking loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
import logging
from tqdm import tqdm
import random
from datetime import datetime
import wandb

from ..models.dihrl_model import DiHRLModel, HierarchicalRankingLoss
from ..models.easy_first_decoder import EasyFirstDecoder
from ..utils.data_utils import Dialogue, IRCDataLoader, collate_dialogues
from ..evaluation.metrics import DialogueDisentanglementMetrics, links_to_clusters


class DialogueDataset(Dataset):
    """Dataset class for dialogue disentanglement training."""
    
    def __init__(self, dialogues: List[Dialogue], tokenizer: BertTokenizer, config: Dict):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.config = config
        self.context_window_size = config['data']['context_window_size']
        self.max_utterance_length = config['data']['max_utterance_length']
        
        # Generate training samples
        self.samples = self._generate_training_samples()
        
    def _generate_training_samples(self) -> List[Dict]:
        """Generate training samples from dialogues."""
        samples = []
        
        for dialogue in self.dialogues:
            for utterance in dialogue.utterances:
                # Find candidate parent utterances
                candidates = []
                for candidate in dialogue.utterances:
                    if (candidate.id <= utterance.id and 
                        utterance.id - candidate.id <= self.context_window_size):
                        candidates.append(candidate.id)
                
                if len(candidates) > 1:  # Need at least 2 candidates for meaningful training
                    # Get ground truth parent
                    ground_truth_parent = dialogue.reply_to.get(utterance.id, utterance.id)
                    
                    sample = {
                        'dialogue': dialogue,
                        'current_utterance_id': utterance.id,
                        'candidate_ids': candidates,
                        'ground_truth_parent': ground_truth_parent
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_training_samples(batch: List[Dict]) -> Dict:
    """Collate function for training DataLoader."""
    return {
        'samples': batch,
        'batch_size': len(batch)
    }


class DiHRLTrainer:
    """Trainer class for DiHRL model with GPU support."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = DiHRLModel(config).to(device)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            config['model']['bert']['model_name']
        )
        
        # Initialize loss function
        training_config = config['training']
        self.criterion = HierarchicalRankingLoss(
            alpha1=training_config['alpha1'],
            alpha2=training_config['alpha2'],
            alpha3=training_config['alpha3']
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = None  # Will be set up during training
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Metrics
        self.metrics_calculator = DialogueDisentanglementMetrics()
        
        # Easy-first decoder for evaluation
        self.decoder = EasyFirstDecoder(self.model, self.tokenizer, config)
        
        # Logging
        self.setup_logging()
        
        # Mixed precision training
        self.use_mixed_precision = config['hardware'].get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = self.config['logging'].get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb if configured
        if self.config['logging'].get('wandb_project'):
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"{self.config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Set up optimizer with different learning rates for BERT and non-BERT parameters."""
        training_config = self.config['training']
        
        # Separate BERT and non-BERT parameters
        bert_params = []
        non_bert_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                non_bert_params.append(param)
        
        # Create parameter groups with different learning rates
        optimizer_params = [
            {
                'params': bert_params,
                'lr': training_config['learning_rate_bert']
            },
            {
                'params': non_bert_params,
                'lr': training_config['learning_rate_non_bert']
            }
        ]
        
        return optim.AdamW(
            optimizer_params,
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    
    def _setup_scheduler(self, num_training_steps: int):
        """Set up learning rate scheduler."""
        warmup_steps = self.config['training'].get('warmup_steps', 1000)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train(self, train_dialogues: List[Dialogue], 
              val_dialogues: List[Dialogue]) -> Dict[str, List[float]]:
        """
        Train the DiHRL model.
        
        Args:
            train_dialogues: Training dialogues
            val_dialogues: Validation dialogues
            
        Returns:
            training_history: Dictionary containing training metrics over time
        """
        self.logger.info("Starting DiHRL training...")
        
        # Create datasets
        train_dataset = DialogueDataset(train_dialogues, self.tokenizer, self.config)
        val_dataset = DialogueDataset(val_dialogues, self.tokenizer, self.config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            collate_fn=collate_training_samples,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            collate_fn=collate_training_samples,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        
        # Set up scheduler
        num_epochs = self.config['training']['num_epochs']
        num_training_steps = len(train_loader) * num_epochs
        self._setup_scheduler(num_training_steps)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader, val_dialogues)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_metrics.get('cluster_f1', 0.0):.4f}"
            )
            
            # Wandb logging
            if wandb.run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
            
            # Save checkpoint
            if self.config['logging'].get('save_checkpoints', True):
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            current_metric = val_metrics.get('cluster_f1', 0.0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                # Save best model
                self._save_best_model()
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['training'].get('early_stopping_patience', 5):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.logger.info("Training completed!")
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Add teacher forcing noise probability
        teacher_forcing_prob = self.config['training'].get('teacher_forcing_noise_prob', 0.15)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_samples = 0
            
            # Process each sample in the batch
            for sample in batch['samples']:
                try:
                    # Add teacher forcing noise to partial replying structure
                    partial_replying_edges = {}
                    if random.random() < teacher_forcing_prob:
                        # Add some noise by including wrong parent relations
                        dialogue = sample['dialogue']
                        for uid, parent in dialogue.reply_to.items():
                            if random.random() < 0.15:  # 15% chance to add noise
                                wrong_parent = random.choice([u.id for u in dialogue.utterances 
                                                            if u.id != parent and u.id <= uid])
                                partial_replying_edges[(uid, wrong_parent)] = 0.5
                            else:
                                partial_replying_edges[(uid, parent)] = 1.0
                    else:
                        # Use correct partial replying structure
                        dialogue = sample['dialogue']
                        for uid, parent in dialogue.reply_to.items():
                            if uid != sample['current_utterance_id']:  # Don't include current utterance
                                partial_replying_edges[(uid, parent)] = 1.0
                    
                    # Forward pass
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            scores = self.model(
                                dialogue=sample['dialogue'],
                                current_utterance_id=sample['current_utterance_id'],
                                candidate_utterance_ids=sample['candidate_ids'],
                                tokenizer=self.tokenizer,
                                partial_replying_edges=partial_replying_edges
                            )
                            
                            # Compute hierarchical ranking loss
                            loss, loss_components = self.criterion(
                                scores=scores,
                                dialogue=sample['dialogue'],
                                current_utterance_id=sample['current_utterance_id'],
                                candidate_ids=sample['candidate_ids'],
                                ground_truth_parent=sample['ground_truth_parent']
                            )
                    else:
                        scores = self.model(
                            dialogue=sample['dialogue'],
                            current_utterance_id=sample['current_utterance_id'],
                            candidate_utterance_ids=sample['candidate_ids'],
                            tokenizer=self.tokenizer,
                            partial_replying_edges=partial_replying_edges
                        )
                        
                        # Compute hierarchical ranking loss
                        loss, loss_components = self.criterion(
                            scores=scores,
                            dialogue=sample['dialogue'],
                            current_utterance_id=sample['current_utterance_id'],
                            candidate_ids=sample['candidate_ids'],
                            ground_truth_parent=sample['ground_truth_parent']
                        )
                    
                    batch_loss += loss
                    batch_samples += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing sample: {e}")
                    continue
            
            if batch_samples > 0:
                # Average loss over batch
                batch_loss = batch_loss / batch_samples
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scaler.scale(batch_loss).backward()
                    
                    # Gradient clipping
                    if self.config['training'].get('gradient_clipping', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clipping']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    batch_loss.backward()
                    
                    # Gradient clipping
                    if self.config['training'].get('gradient_clipping', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['gradient_clipping']
                        )
                    
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': batch_loss.item()})
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, val_loader: DataLoader, val_dialogues: List[Dialogue]) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Collect predictions for evaluation
        all_predicted_links = {}
        all_gold_links = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_loss = 0.0
                batch_samples = 0
                
                for sample in batch['samples']:
                    try:
                        # Forward pass without teacher forcing
                        scores = self.model(
                            dialogue=sample['dialogue'],
                            current_utterance_id=sample['current_utterance_id'],
                            candidate_utterance_ids=sample['candidate_ids'],
                            tokenizer=self.tokenizer,
                            partial_replying_edges={}  # Empty for validation
                        )
                        
                        # Compute loss
                        loss, _ = self.criterion(
                            scores=scores,
                            dialogue=sample['dialogue'],
                            current_utterance_id=sample['current_utterance_id'],
                            candidate_ids=sample['candidate_ids'],
                            ground_truth_parent=sample['ground_truth_parent']
                        )
                        
                        batch_loss += loss
                        batch_samples += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing validation sample: {e}")
                        continue
                
                if batch_samples > 0:
                    total_loss += (batch_loss / batch_samples).item()
                    num_batches += 1
        
        # Evaluate on full dialogues using easy-first decoding
        for dialogue in val_dialogues[:min(10, len(val_dialogues))]:  # Limit for speed
            try:
                predicted_links = self.decoder.decode_dialogue(dialogue)
                
                dialogue_id = dialogue.id
                all_predicted_links[dialogue_id] = predicted_links
                all_gold_links[dialogue_id] = dialogue.reply_to
                
            except Exception as e:
                self.logger.warning(f"Error decoding dialogue {dialogue.id}: {e}")
                continue
        
        # Compute metrics
        if all_predicted_links and all_gold_links:
            # Flatten predictions across all dialogues
            flat_pred_links = {}
            flat_gold_links = {}
            
            for dialogue_id in all_predicted_links:
                flat_pred_links.update(all_predicted_links[dialogue_id])
                flat_gold_links.update(all_gold_links[dialogue_id])
            
            # Convert to clusters
            pred_clusters = links_to_clusters(flat_pred_links)
            gold_clusters = links_to_clusters(flat_gold_links)
            
            # Compute metrics
            metrics = self.metrics_calculator.evaluate_all(
                predicted_clusters=pred_clusters,
                gold_clusters=gold_clusters,
                predicted_links=flat_pred_links,
                gold_links=flat_gold_links
            )
        else:
            metrics = {}
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = self.config['logging'].get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch + 1}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
    
    def _save_best_model(self):
        """Save the best model."""
        checkpoint_dir = self.config['logging'].get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, best_model_path)
        
        self.logger.info(f"Saved best model to {best_model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")


def setup_gpu_training(config: Dict) -> torch.device:
    """Set up GPU training environment."""
    # Set device
    if torch.cuda.is_available() and config['hardware']['device'] == 'cuda':
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f"GPU training enabled. Available GPUs: {num_gpus}")
        
        # Set GPU memory allocation strategy
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            
        # Set deterministic behavior if requested
        if config['experiment'].get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')
        print("Using CPU for training")
    
    # Set random seeds for reproducibility
    seed = config['experiment'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return device
