"""
Main DiHRL model implementation.
Combines BERT encoding, discourse structure modeling, BiLSTM, and prediction layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple, Optional, Set
import math
import numpy as np

from .egcn import DiscourseStructureEncoder, DropConnectLSTM
from .discourse_structures import DiscourseStructureBuilder, DiscourseGraph
from ..utils.data_utils import Dialogue, Utterance


class DiHRLModel(nn.Module):
    """
    Main DiHRL model for dialogue disentanglement.
    Implements the structure-aware framework with hierarchical ranking loss.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(config['model']['bert']['model_name'])
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Discourse structure builder
        self.discourse_builder = DiscourseStructureBuilder(config['discourse_structures'])
        
        # EGCN configuration
        egcn_config = {
            'input_dim': self.bert_hidden_size,
            'hidden_dims': config['model']['egcn']['hidden_sizes'],
            'edge_label_dim': config['model']['egcn']['edge_label_embedding_dim'],
            'activation': config['model']['egcn']['activation'],
            'dropout': config['model']['egcn']['dropout'],
            'fusion_method': 'addition'  # As specified in paper
        }
        
        # Discourse structure encoder
        self.discourse_encoder = DiscourseStructureEncoder(egcn_config)
        
        # BiLSTM with DropConnect
        lstm_config = config['model']['lstm']
        self.lstm = DropConnectLSTM(
            input_size=egcn_config['hidden_dims'][-1],
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout_feedforward=lstm_config['dropout_feedforward'],
            dropout_recurrent=lstm_config['dropout_recurrent'],
            bidirectional=lstm_config['bidirectional']
        )
        
        # Final hidden size after BiLSTM
        lstm_output_size = lstm_config['hidden_size'] * (2 if lstm_config['bidirectional'] else 1)
        
        # Feedforward network for prediction
        ffnn_config = config['model']['ffnn']
        self.ffnn = nn.Sequential(
            nn.Linear(lstm_output_size + self.bert_hidden_size, ffnn_config['layer1_hidden_size']),
            nn.ReLU() if ffnn_config['activation'] == 'relu' else nn.Tanh(),
            nn.Dropout(ffnn_config['dropout']),
            nn.Linear(ffnn_config['layer1_hidden_size'], ffnn_config['layer2_hidden_size']),
            nn.ReLU() if ffnn_config['activation'] == 'relu' else nn.Tanh(),
            nn.Dropout(ffnn_config['dropout']),
            nn.Linear(ffnn_config['layer2_hidden_size'], 1)
        )
        
        # Training configuration
        self.training_config = config['training']
        self.context_window_size = config['data']['context_window_size']
        
        # Dropout
        self.dropout = nn.Dropout(config['model']['bert']['dropout'])
        
    def encode_utterance_pairs(self, tokenizer: BertTokenizer, current_utterance: Utterance,
                              candidate_utterances: List[Utterance], max_length: int = 128) -> torch.Tensor:
        """
        Encode utterance pairs using BERT.
        
        Args:
            tokenizer: BERT tokenizer
            current_utterance: Current utterance to find parent for
            candidate_utterances: List of candidate parent utterances
            max_length: Maximum sequence length
            
        Returns:
            pair_representations: [num_candidates, hidden_size] - BERT representations
        """
        device = next(self.parameters()).device
        
        # Prepare input pairs
        input_texts = []
        for candidate in candidate_utterances:
            # Format: [CLS] candidate_text [SEP] current_text [SEP]
            text = f"{candidate.text} [SEP] {current_utterance.text}"
            input_texts.append(text)
        
        # Tokenize
        encoded = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Encode with BERT
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use [CLS] token representation
            pair_representations = outputs.last_hidden_state[:, 0, :]  # [num_candidates, hidden_size]
        
        return self.dropout(pair_representations)
    
    def forward(self, dialogue: Dialogue, current_utterance_id: int,
                candidate_utterance_ids: List[int], tokenizer: BertTokenizer,
                partial_replying_edges: Optional[Dict[Tuple[int, int], float]] = None) -> torch.Tensor:
        """
        Forward pass of DiHRL model.
        
        Args:
            dialogue: Complete dialogue object
            current_utterance_id: ID of current utterance to find parent for
            candidate_utterance_ids: List of candidate parent utterance IDs
            tokenizer: BERT tokenizer
            partial_replying_edges: Established partial replying relations
            
        Returns:
            scores: [num_candidates] - Parent prediction scores
        """
        device = next(self.parameters()).device
        
        # Get utterances
        current_utterance = next(u for u in dialogue.utterances if u.id == current_utterance_id)
        candidate_utterances = [next(u for u in dialogue.utterances if u.id == uid) 
                              for uid in candidate_utterance_ids]
        
        # Define context window
        min_candidate_id = min(candidate_utterance_ids)
        max_candidate_id = max([current_utterance_id] + candidate_utterance_ids)
        context_window = (max(0, min_candidate_id - self.context_window_size), 
                         max_candidate_id + self.context_window_size)
        
        # 1. Utterance Encoding with BERT
        pair_representations = self.encode_utterance_pairs(
            tokenizer, current_utterance, candidate_utterances
        )  # [num_candidates, bert_hidden_size]
        
        # 2. Build discourse structures for context window
        discourse_graphs = self.discourse_builder.build_all_structures(
            dialogue, context_window, partial_replying_edges
        )
        
        # Create node features for context window utterances
        context_utterances = [u for u in dialogue.utterances 
                            if context_window[0] <= u.id <= context_window[1]]
        
        # Map candidate representations to context window
        context_node_features = torch.zeros(len(context_utterances), self.bert_hidden_size, device=device)
        utterance_id_to_context_idx = {u.id: i for i, u in enumerate(context_utterances)}
        
        for i, candidate_id in enumerate(candidate_utterance_ids):
            if candidate_id in utterance_id_to_context_idx:
                context_idx = utterance_id_to_context_idx[candidate_id]
                context_node_features[context_idx] = pair_representations[i]
        
        # Handle current utterance representation
        if current_utterance_id in utterance_id_to_context_idx:
            current_idx = utterance_id_to_context_idx[current_utterance_id]
            # Use average of candidate representations as approximation
            context_node_features[current_idx] = pair_representations.mean(dim=0)
        
        # Fill in missing utterances with zero vectors (will be learned)
        
        # 3. Discourse Structure Modeling with EGCN
        discourse_features = self.discourse_encoder(context_node_features, discourse_graphs)
        
        # 4. Extract features for candidates
        candidate_discourse_features = []
        for candidate_id in candidate_utterance_ids:
            if candidate_id in utterance_id_to_context_idx:
                context_idx = utterance_id_to_context_idx[candidate_id]
                candidate_discourse_features.append(discourse_features[context_idx])
            else:
                # Use zero vector for out-of-context candidates
                candidate_discourse_features.append(torch.zeros_like(discourse_features[0]))
        
        candidate_discourse_features = torch.stack(candidate_discourse_features)  # [num_candidates, egcn_hidden_size]
        
        # 5. Temporal modeling with BiLSTM
        # Prepare sequence: arrange candidates by chronological order
        sorted_indices = sorted(range(len(candidate_utterance_ids)), 
                              key=lambda i: candidate_utterance_ids[i])
        
        # Create sequence for LSTM
        lstm_input = candidate_discourse_features[sorted_indices].unsqueeze(0)  # [1, num_candidates, hidden_size]
        lstm_output, _ = self.lstm(lstm_input)  # [1, num_candidates, lstm_hidden_size]
        lstm_output = lstm_output.squeeze(0)  # [num_candidates, lstm_hidden_size]
        
        # Reorder back to original candidate order
        reorder_indices = [0] * len(sorted_indices)
        for i, orig_idx in enumerate(sorted_indices):
            reorder_indices[orig_idx] = i
        
        temporal_features = lstm_output[reorder_indices]  # [num_candidates, lstm_hidden_size]
        
        # 6. Residual connection with original BERT features
        combined_features = torch.cat([temporal_features, pair_representations], dim=1)
        
        # 7. Final prediction
        scores = self.ffnn(combined_features).squeeze(-1)  # [num_candidates]
        
        return scores
    
    def predict_parent(self, dialogue: Dialogue, current_utterance_id: int,
                      tokenizer: BertTokenizer,
                      partial_replying_edges: Optional[Dict[Tuple[int, int], float]] = None) -> Tuple[int, torch.Tensor]:
        """
        Predict the parent utterance for a given current utterance.
        
        Args:
            dialogue: Complete dialogue object
            current_utterance_id: ID of current utterance
            tokenizer: BERT tokenizer
            partial_replying_edges: Established partial replying relations
            
        Returns:
            parent_id: Predicted parent utterance ID
            scores: All candidate scores
        """
        # Find candidate parent utterances (precedent utterances + self)
        candidate_ids = []
        for utterance in dialogue.utterances:
            if utterance.id <= current_utterance_id:
                # Within context window
                if current_utterance_id - utterance.id <= self.context_window_size:
                    candidate_ids.append(utterance.id)
        
        if not candidate_ids:
            # No valid candidates, return self as parent
            return current_utterance_id, torch.tensor([1.0])
        
        # Get prediction scores
        with torch.no_grad():
            scores = self.forward(dialogue, current_utterance_id, candidate_ids, 
                                tokenizer, partial_replying_edges)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(scores, dim=0)
        
        # Select best candidate
        best_idx = torch.argmax(probabilities).item()
        parent_id = candidate_ids[best_idx]
        
        return parent_id, probabilities


class HierarchicalRankingLoss(nn.Module):
    """
    Hierarchical Ranking Loss as described in the paper.
    Implements L1, L2, and L3 losses for different discourse levels.
    """
    
    def __init__(self, alpha1: float = 1.0, alpha2: float = 0.1, alpha3: float = 0.05):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
    
    def categorize_candidates(self, dialogue: Dialogue, current_utterance_id: int,
                            candidate_ids: List[int], ground_truth_parent: int) -> Dict[str, List[int]]:
        """
        Categorize candidate utterances into discourse levels R1, R2, R3, R4.
        
        Args:
            dialogue: Complete dialogue object
            current_utterance_id: Current utterance ID
            candidate_ids: List of candidate parent IDs
            ground_truth_parent: True parent utterance ID
            
        Returns:
            categories: Dictionary mapping level names to utterance ID lists
        """
        categories = {'R1': [], 'R2': [], 'R3': [], 'R4': []}
        
        # R1: Parent utterance (ground truth)
        categories['R1'] = [ground_truth_parent] if ground_truth_parent in candidate_ids else []
        
        # Find the session/thread for current utterance
        current_session = dialogue.get_session_for_utterance(current_utterance_id)
        
        # R2: Ancestor utterances (same thread, but not direct parent)
        # Find utterances in same thread that are ancestors
        for candidate_id in candidate_ids:
            if candidate_id != ground_truth_parent and candidate_id in current_session:
                # Check if it's an ancestor (path exists from candidate to current through parent)
                if self._is_ancestor(dialogue, candidate_id, current_utterance_id, ground_truth_parent):
                    categories['R2'].append(candidate_id)
        
        # R3: Inner-session utterances (same session but different thread)
        for candidate_id in candidate_ids:
            if (candidate_id not in categories['R1'] and 
                candidate_id not in categories['R2']):
                
                candidate_session = dialogue.get_session_for_utterance(candidate_id)
                
                # Check if they share any utterances (same session)
                if set(current_session) & set(candidate_session):
                    categories['R3'].append(candidate_id)
        
        # R4: Outer-session utterances (different sessions)
        for candidate_id in candidate_ids:
            if (candidate_id not in categories['R1'] and 
                candidate_id not in categories['R2'] and 
                candidate_id not in categories['R3']):
                categories['R4'].append(candidate_id)
        
        return categories
    
    def _is_ancestor(self, dialogue: Dialogue, candidate_id: int, 
                    current_id: int, parent_id: int) -> bool:
        """Check if candidate_id is an ancestor of current_id."""
        # Simple heuristic: ancestor if it's in the reply chain leading to parent
        current = parent_id
        while current in dialogue.reply_to and dialogue.reply_to[current] != current:
            current = dialogue.reply_to[current]
            if current == candidate_id:
                return True
        return False
    
    def forward(self, scores: torch.Tensor, dialogue: Dialogue, 
                current_utterance_id: int, candidate_ids: List[int],
                ground_truth_parent: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hierarchical ranking loss.
        
        Args:
            scores: [num_candidates] - prediction scores
            dialogue: Complete dialogue object
            current_utterance_id: Current utterance ID
            candidate_ids: List of candidate parent IDs
            ground_truth_parent: True parent utterance ID
            
        Returns:
            total_loss: Combined hierarchical loss
            loss_components: Dictionary of individual loss components
        """
        # Categorize candidates
        categories = self.categorize_candidates(dialogue, current_utterance_id, 
                                              candidate_ids, ground_truth_parent)
        
        # Create index mappings
        id_to_idx = {uid: i for i, uid in enumerate(candidate_ids)}
        
        loss_components = {}
        
        # L1 Loss: Parent vs all others
        if categories['R1']:
            r1_indices = [id_to_idx[uid] for uid in categories['R1'] if uid in id_to_idx]
            if r1_indices:
                r1_scores = scores[r1_indices]
                all_scores = scores
                
                # Softmax with R1 in numerator, all candidates in denominator
                l1_loss = -torch.log(torch.sum(torch.exp(r1_scores)) / torch.sum(torch.exp(all_scores)))
                loss_components['L1'] = l1_loss
            else:
                loss_components['L1'] = torch.tensor(0.0, device=scores.device)
        else:
            loss_components['L1'] = torch.tensor(0.0, device=scores.device)
        
        # L2 Loss: Ancestors vs {ancestors, inner-session, outer-session}
        if categories['R2']:
            r2_indices = [id_to_idx[uid] for uid in categories['R2'] if uid in id_to_idx]
            r234_indices = []
            for level in ['R2', 'R3', 'R4']:
                r234_indices.extend([id_to_idx[uid] for uid in categories[level] if uid in id_to_idx])
            
            if r2_indices and r234_indices:
                r2_scores = scores[r2_indices]
                r234_scores = scores[r234_indices]
                
                l2_loss = -torch.log(torch.sum(torch.exp(r2_scores)) / torch.sum(torch.exp(r234_scores)))
                loss_components['L2'] = l2_loss
            else:
                loss_components['L2'] = torch.tensor(0.0, device=scores.device)
        else:
            loss_components['L2'] = torch.tensor(0.0, device=scores.device)
        
        # L3 Loss: Inner-session vs {inner-session, outer-session}
        if categories['R3']:
            r3_indices = [id_to_idx[uid] for uid in categories['R3'] if uid in id_to_idx]
            r34_indices = []
            for level in ['R3', 'R4']:
                r34_indices.extend([id_to_idx[uid] for uid in categories[level] if uid in id_to_idx])
            
            if r3_indices and r34_indices:
                r3_scores = scores[r3_indices]
                r34_scores = scores[r34_indices]
                
                l3_loss = -torch.log(torch.sum(torch.exp(r3_scores)) / torch.sum(torch.exp(r34_scores)))
                loss_components['L3'] = l3_loss
            else:
                loss_components['L3'] = torch.tensor(0.0, device=scores.device)
        else:
            loss_components['L3'] = torch.tensor(0.0, device=scores.device)
        
        # Combine losses
        total_loss = (self.alpha1 * loss_components['L1'] + 
                     self.alpha2 * loss_components['L2'] + 
                     self.alpha3 * loss_components['L3'])
        
        return total_loss, loss_components
