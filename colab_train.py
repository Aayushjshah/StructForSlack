#!/usr/bin/env python3
"""
Colab-Optimized Training Script for IRC Conversation Disentanglement
Supports GPU/TPU training with advanced memory management
"""

import os
import sys
import gc
import json
import pickle
import logging
import argparse
import warnings
import urllib.request
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

# Transformers imports
from transformers import (
    BertConfig, BertTokenizer, BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    set_seed, TrainingArguments, Trainer
)

# Memory and performance optimization
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ColabConfig:
    """Configuration class for Colab environment optimization"""
    
    def __init__(self):
        self.detect_environment()
        self.configure_memory_settings()
        
    def detect_environment(self):
        """Detect available hardware and set optimal configurations"""
        self.is_colab = 'google.colab' in sys.modules
        self.device_type = 'cpu'
        self.device_count = 0
        self.available_memory = 0
        
        # Detect TPU
        try:
            import torch_xla.core.xla_model as xm
            self.device_type = 'tpu'
            self.device = xm.xla_device()
            self.device_count = xm.xrt_world_size()
            logger.info(f"TPU detected with {self.device_count} cores")
        except ImportError:
            pass
        
        # Detect CUDA GPU
        if self.device_type == 'cpu' and torch.cuda.is_available():
            self.device_type = 'cuda'
            self.device = torch.device('cuda')
            self.device_count = torch.cuda.device_count()
            self.available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {self.available_memory:.1f} GB")
        
        if self.device_type == 'cpu':
            self.device = torch.device('cpu')
            logger.info("Using CPU for training")
    
    def configure_memory_settings(self):
        """Configure memory-optimized settings based on available hardware"""
        if self.device_type == 'cuda':
            # GPU memory optimization
            if self.available_memory < 8:  # Less than 8GB
                self.max_seq_length = 128
                self.train_batch_size = 1
                self.eval_batch_size = 2
                self.gradient_accumulation_steps = 8
            elif self.available_memory < 16:  # 8-16GB
                self.max_seq_length = 256
                self.train_batch_size = 2
                self.eval_batch_size = 4
                self.gradient_accumulation_steps = 4
            else:  # 16GB+
                self.max_seq_length = 512
                self.train_batch_size = 4
                self.eval_batch_size = 8
                self.gradient_accumulation_steps = 2
        elif self.device_type == 'tpu':
            # TPU optimization
            self.max_seq_length = 256
            self.train_batch_size = 8
            self.eval_batch_size = 8
            self.gradient_accumulation_steps = 1
        else:
            # CPU fallback
            self.max_seq_length = 128
            self.train_batch_size = 1
            self.eval_batch_size = 1
            self.gradient_accumulation_steps = 4
        
        logger.info(f"Configured settings - Seq length: {self.max_seq_length}, "
                   f"Train batch: {self.train_batch_size}, "
                   f"Grad accumulation: {self.gradient_accumulation_steps}")

class MemoryOptimizedBertV2(BertPreTrainedModel):
    """Memory-optimized version of Bert_v2 for Colab training"""
    
    def __init__(self, config, lstm_hidden_size=64, lstm_num_layers=1, 
                 gcn_layer=1, mylstm_hidden_size=64, num_decoupling=1):
        super().__init__(config)
        
        # Reduced model dimensions for memory efficiency
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_decoupling = num_decoupling
        self.gcn_layer = gcn_layer
        self.mylstm_hidden_size = mylstm_hidden_size
        self.graph_dim = config.hidden_size
        
        # Core components
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Simplified architecture for memory efficiency
        self.pooler = nn.Linear(2 * self.mylstm_hidden_size, self.mylstm_hidden_size)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Linear(self.mylstm_hidden_size, 1)
        
        # Graph components
        self.W = nn.ModuleList()
        for layer in range(self.gcn_layer):
            self.W.append(nn.Linear(self.graph_dim, self.graph_dim))
        
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, adj_matrix_speaker=None, adj_matrix_mention=None, **kwargs):
        
        # Use gradient checkpointing for memory efficiency
        if self.training:
            self.bert.encoder.gradient_checkpointing = True
        
        num_labels = input_ids.shape[1] if input_ids is not None else 1
        
        # Reshape inputs
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        # BERT forward pass with memory optimization
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False
        )
        
        sequence_output = outputs[0]  # (batch_size * num_choice, seq_len, hidden_size)
        cls_rep = sequence_output[:, 0, :]  # CLS token representation
        hidden_size = sequence_output.size(-1)
        cls_rep = cls_rep.view(-1, num_labels, hidden_size)
        
        # Simplified processing for memory efficiency
        if adj_matrix_mention is not None:
            adj_matrix = adj_matrix_mention.float()
            batch_size, sent_len, input_dim = cls_rep.size()
            
            # Graph convolution (simplified)
            if self.gcn_layer > 0 and adj_matrix.size(1) == sent_len:
                denom = adj_matrix.sum(2).unsqueeze(2) + 1
                graph_input = cls_rep
                
                for l in range(self.gcn_layer):
                    Ax = adj_matrix.bmm(graph_input)
                    AxW = self.W[l](Ax) + self.W[l](graph_input)
                    AxW = AxW / denom
                    graph_input = torch.relu(AxW)
                
                # Simplified combination
                combined_rep = torch.cat([cls_rep, graph_input], dim=-1)
                pooled_output = self.pooler_activation(self.pooler(combined_rep))
            else:
                # Fallback without graph processing
                pooled_output = cls_rep
        else:
            pooled_output = cls_rep
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(-1)
        
        outputs = (logits,)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        
        return outputs

class IRCDataProcessor:
    """Optimized data processor for IRC disentanglement"""
    
    def __init__(self, data_dir: str, max_previous_utterance: int = 3):
        self.data_dir = data_dir
        self.max_previous_utterance = max_previous_utterance
        
    def get_examples(self, mode: str, limit: Optional[int] = None):
        """Load and process examples with optional limiting for testing"""
        logger.info(f"Loading {mode} examples from {self.data_dir}")
        
        mode_dir = os.path.join(self.data_dir, mode)
        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"Directory {mode_dir} not found")
        
        examples = []
        files = os.listdir(mode_dir)
        ascii_files = [f for f in files if f.endswith('.ascii.txt')]
        
        if limit:
            ascii_files = ascii_files[:limit]
        
        for file in tqdm(ascii_files, desc=f"Processing {mode} files"):
            file_path = os.path.join(mode_dir, file)
            try:
                text_ascii = [line.strip().split() for line in open(file_path)]
                
                # Read labels
                label_file = file_path.replace('.ascii.txt', '.annotation.txt')
                if os.path.exists(label_file):
                    labels = [line.strip().split() for line in open(label_file)]
                    examples.append({
                        'filename': file.split('.')[0],
                        'text_ascii': text_ascii,
                        'labels': labels
                    })
            except Exception as e:
                logger.warning(f"Error processing {file}: {e}")
                continue
        
        return self._reshape_examples(examples, mode)
    
    def _reshape_examples(self, examples, mode):
        """Convert examples to trainable format"""
        reshaped_examples = []
        
        for example in tqdm(examples, desc="Reshaping examples"):
            if not example['labels']:
                continue
                
            main_text_indices = [int(label[1]) for label in example['labels']]
            if not main_text_indices:
                continue
                
            begin, end = min(main_text_indices), max(main_text_indices)
            
            for i in range(begin, min(end + 1, len(example['text_ascii']))):
                try:
                    # Generate training example
                    text_a, text_b = [], []
                    
                    # Get label (distance to nearest reference)
                    label = 0
                    for label_line in example['labels']:
                        if int(label_line[1]) == i:
                            label = i - int(label_line[0])
                            break
                    
                    if label >= self.max_previous_utterance:
                        label = 0
                    
                    # Create context pairs
                    for j in range(self.max_previous_utterance):
                        if i - j >= 0 and i - j < len(example['text_ascii']):
                            text_b.append(' '.join(example['text_ascii'][i - j]))
                            text_a.append(' '.join(example['text_ascii'][i]))
                        else:
                            text_b.append('[EMPTY]')
                            text_a.append('[EMPTY]')
                    
                    # Simple adjacency matrices (placeholder)
                    adj_matrix = [[0] * self.max_previous_utterance for _ in range(self.max_previous_utterance)]
                    
                    reshaped_examples.append({
                        'guid': f"{example['filename']}_{i}",
                        'text_a': text_a,
                        'text_b': text_b,
                        'label': label,
                        'adj_matrix_speaker': adj_matrix,
                        'adj_matrix_mention': adj_matrix
                    })
                    
                except Exception as e:
                    logger.warning(f"Error reshaping example {i}: {e}")
                    continue
        
        return reshaped_examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, max_utterance_num):
    """Convert examples to model features with memory optimization"""
    features = []
    
    for ex_index, example in enumerate(tqdm(examples, desc="Converting examples")):
        if ex_index % 1000 == 0:
            logger.info(f"Processing example {ex_index}/{len(examples)}")
        
        try:
            choices_features = []
            
            for text_a, text_b in zip(example['text_a'], example['text_b']):
                # Tokenize with error handling
                try:
                    tokens_a = tokenizer.tokenize(str(text_a))[:max_seq_length//2]
                    tokens_b = tokenizer.tokenize(str(text_b))[:max_seq_length//2]
                except:
                    tokens_a = ['[UNK]']
                    tokens_b = ['[UNK]']
                
                # Build sequence
                tokens = ['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]']
                tokens = tokens[:max_seq_length]
                
                # Convert to IDs
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                attention_mask = [1] * len(input_ids)
                token_type_ids = ([0] * (len(tokens_b) + 2) + 
                                [1] * (len(tokens) - len(tokens_b) - 2))
                
                # Padding
                padding_length = max_seq_length - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                token_type_ids += [0] * padding_length
                
                choices_features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids
                })
            
            features.append({
                'guid': example['guid'],
                'choices_features': choices_features,
                'label': example['label'],
                'adj_matrix_speaker': example['adj_matrix_speaker'],
                'adj_matrix_mention': example['adj_matrix_mention']
            })
            
        except Exception as e:
            logger.warning(f"Error converting example {ex_index}: {e}")
            continue
    
    return features

def create_data_loader(features, batch_size, is_training=False):
    """Create optimized data loader"""
    try:
        # Extract tensors
        all_input_ids = torch.tensor([f['choices_features'][0]['input_ids'] for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f['choices_features'][0]['attention_mask'] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f['choices_features'][0]['token_type_ids'] for f in features], dtype=torch.long)
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
        
        # Create adjacency matrices (simplified)
        max_choices = max(len(f['choices_features']) for f in features)
        all_adj_speaker = torch.zeros(len(features), max_choices, max_choices, dtype=torch.long)
        all_adj_mention = torch.zeros(len(features), max_choices, max_choices, dtype=torch.long)
        
        for i, f in enumerate(features):
            size = len(f['adj_matrix_speaker'])
            if size > 0:
                adj_speaker = torch.tensor(f['adj_matrix_speaker'][:size][:size], dtype=torch.long)
                adj_mention = torch.tensor(f['adj_matrix_mention'][:size][:size], dtype=torch.long)
                all_adj_speaker[i, :size, :size] = adj_speaker
                all_adj_mention[i, :size, :size] = adj_mention
        
        # Reshape for multiple choice format
        batch_size_total = all_input_ids.size(0)
        all_input_ids = all_input_ids.unsqueeze(1)  # Add choice dimension
        all_attention_mask = all_attention_mask.unsqueeze(1)
        all_token_type_ids = all_token_type_ids.unsqueeze(1)
        
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_labels, all_adj_speaker, all_adj_mention
        )
        
        sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)
        
        return dataloader
        
    except Exception as e:
        logger.error(f"Error creating data loader: {e}")
        raise

def train_model(model, train_loader, eval_loader, config, device):
    """Main training loop with memory optimization"""
    
    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-8,
        weight_decay=0.01
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Mixed precision scaler for GPU
    scaler = GradScaler() if config.device_type == 'cuda' else None
    
    # Training tracking
    train_losses = []
    eval_accuracies = []
    
    model.to(device)
    
    for epoch in range(config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
        
        for step, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                batch = tuple(t.to(device) for t in batch)
                
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3],
                    'adj_matrix_speaker': batch[4],
                    'adj_matrix_mention': batch[5]
                }
                
                # Forward pass with mixed precision
                if scaler:
                    with autocast():
                        outputs = model(**inputs)
                        loss = outputs[0]
                        loss = loss / config.gradient_accumulation_steps
                else:
                    outputs = model(**inputs)
                    loss = outputs[0]
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_train_loss += loss.item()
                
                # Optimizer step
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Memory cleanup
                    if config.device_type == 'cuda':
                        torch.cuda.empty_cache()
                
                num_batches += 1
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_train_loss/num_batches:.4f}'
                })
                
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                continue
        
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        if eval_loader:
            eval_accuracy = evaluate_model(model, eval_loader, device, config)
            eval_accuracies.append(eval_accuracy)
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
        else:
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_train_loss, config)
        
        # Memory cleanup
        gc.collect()
        if config.device_type == 'cuda':
            torch.cuda.empty_cache()
    
    return train_losses, eval_accuracies

def evaluate_model(model, eval_loader, device, config):
    """Evaluate model performance"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            try:
                batch = tuple(t.to(device) for t in batch)
                
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3],
                    'adj_matrix_speaker': batch[4],
                    'adj_matrix_mention': batch[5]
                }
                
                with autocast() if config.device_type == 'cuda' else torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == batch[3]).sum().item()
                total_samples += batch[3].size(0)
                
            except Exception as e:
                logger.warning(f"Error in evaluation step: {e}")
                continue
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy

def save_checkpoint(model, optimizer, epoch, loss, config):
    """Save model checkpoint"""
    checkpoint_dir = Path(config.output_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config.__dict__
    }
    
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def download_and_extract_data(target_dir="./", force_download=False):
    """Download and extract IRC disentanglement dataset from GitHub"""
    
    data_url = "https://github.com/jkkummerfeld/irc-disentanglement/tarball/master"
    extracted_dir_pattern = "jkkummerfeld-irc-disentanglement-*"
    expected_data_dir = None
    
    # Check if data already exists
    existing_dirs = list(Path(target_dir).glob(extracted_dir_pattern))
    if existing_dirs and not force_download:
        expected_data_dir = str(existing_dirs[0])
        logger.info(f"Data directory already exists: {expected_data_dir}")
        return expected_data_dir
    
    logger.info("Downloading IRC disentanglement dataset...")
    
    try:
        # Create target directory
        Path(target_dir).mkdir(exist_ok=True)
        
        # Download the tarball
        tarball_path = Path(target_dir) / "irc-disentanglement-master.tar.gz"
        
        def download_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 50 == 0:  # Print every 50 blocks to avoid spam
                    logger.info(f"Download progress: {percent}%")
        
        logger.info(f"Downloading from {data_url}")
        urllib.request.urlretrieve(data_url, tarball_path, download_progress)
        logger.info("Download completed!")
        
        # Extract the tarball
        logger.info("Extracting dataset...")
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(target_dir)
        
        # Find the extracted directory
        extracted_dirs = list(Path(target_dir).glob(extracted_dir_pattern))
        if not extracted_dirs:
            raise FileNotFoundError("Could not find extracted directory")
        
        expected_data_dir = str(extracted_dirs[0])
        logger.info(f"Dataset extracted to: {expected_data_dir}")
        
        # Clean up tarball
        tarball_path.unlink()
        logger.info("Cleanup completed")
        
        return expected_data_dir
        
    except Exception as e:
        logger.error(f"Failed to download/extract dataset: {e}")
        raise

def plot_training_progress(train_losses, eval_accuracies, config):
    """Plot training progress"""
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        if eval_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(eval_accuracies)
            plt.title('Evaluation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plot_path = Path(config.output_dir) / 'training_progress.png'
        plt.savefig(plot_path)
        plt.show()
        logger.info(f"Training progress plot saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Could not create training plot: {e}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Colab-optimized IRC Disentanglement Training")
    
    # Data arguments
    parser.add_argument("--data_dir", default="jkkummerfeld-irc-disentanglement-82ed04f/data/", 
                       help="Path to data directory")
    parser.add_argument("--output_dir", default="./colab_output/", 
                       help="Output directory for models and logs")
    parser.add_argument("--cache_dir", default="./cached_models/", 
                       help="Cache directory for pre-trained models")
    
    # Model arguments
    parser.add_argument("--model_name", default="bert-base-uncased", 
                       help="Pre-trained model name")
    parser.add_argument("--max_previous_utterance", type=int, default=3, 
                       help="Maximum previous utterances to consider")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Testing arguments
    parser.add_argument("--limit_examples", type=int, default=None, 
                       help="Limit number of examples for testing (None for all)")
    parser.add_argument("--quick_test", action='store_true', 
                       help="Run quick test with limited data")
    
    # Data download arguments
    parser.add_argument("--auto_download", action='store_true', default=True,
                       help="Automatically download dataset if not found")
    parser.add_argument("--force_download", action='store_true', 
                       help="Force re-download of dataset even if it exists")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    colab_config = ColabConfig()
    
    # Override config with parsed arguments
    colab_config.output_dir = args.output_dir
    colab_config.cache_dir = args.cache_dir
    colab_config.num_epochs = args.num_epochs
    colab_config.learning_rate = args.learning_rate
    
    # Quick test mode
    if args.quick_test:
        args.limit_examples = 100
        colab_config.num_epochs = 1
        logger.info("Running in quick test mode")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    logger.info("Starting Colab-optimized training")
    logger.info(f"Device: {colab_config.device_type}")
    logger.info(f"Batch size: {colab_config.train_batch_size}")
    logger.info(f"Max sequence length: {colab_config.max_seq_length}")
    
    try:
        # Handle data download if needed
        if args.auto_download or args.force_download:
            if not os.path.exists(args.data_dir) or args.force_download:
                logger.info("Data directory not found or force download requested. Downloading dataset...")
                try:
                    extracted_dir = download_and_extract_data("./", args.force_download)
                    args.data_dir = os.path.join(extracted_dir, "data")
                    logger.info(f"Updated data directory to: {args.data_dir}")
                except Exception as e:
                    logger.error(f"Failed to download data: {e}")
                    logger.info("Attempting to use existing data directory...")
        
        # Verify data directory exists
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory {args.data_dir} not found!")
            logger.info("Please either:")
            logger.info("1. Use --auto_download to download the dataset automatically")
            logger.info("2. Manually download and extract the dataset")
            logger.info("3. Provide the correct --data_dir path")
            raise FileNotFoundError(f"Data directory {args.data_dir} not found")
        
        # Initialize tokenizer and model
        logger.info("Loading tokenizer and model...")
        tokenizer = BertTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            do_lower_case=True
        )
        
        config = BertConfig.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            num_labels=args.max_previous_utterance
        )
        
        model = MemoryOptimizedBertV2(config)
        
        # Load data
        logger.info("Loading and processing data...")
        processor = IRCDataProcessor(args.data_dir, args.max_previous_utterance)
        
        train_examples = processor.get_examples('train', limit=args.limit_examples)
        logger.info(f"Loaded {len(train_examples)} training examples")
        
        eval_examples = processor.get_examples('dev', limit=args.limit_examples)
        logger.info(f"Loaded {len(eval_examples)} evaluation examples")
        
        # Convert to features
        logger.info("Converting examples to features...")
        train_features = convert_examples_to_features(
            train_examples, tokenizer, colab_config.max_seq_length, args.max_previous_utterance
        )
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, colab_config.max_seq_length, args.max_previous_utterance
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader = create_data_loader(train_features, colab_config.train_batch_size, is_training=True)
        eval_loader = create_data_loader(eval_features, colab_config.eval_batch_size, is_training=False)
        
        # Train model
        logger.info("Starting training...")
        train_losses, eval_accuracies = train_model(
            model, train_loader, eval_loader, colab_config, colab_config.device
        )
        
        # Plot results
        plot_training_progress(train_losses, eval_accuracies, colab_config)
        
        # Save final model
        final_model_path = Path(args.output_dir) / 'final_model.pt'
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        gc.collect()
        if colab_config.device_type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
