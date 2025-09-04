"""
Edge-Aware Graph Convolutional Network (EGCN) for DiHRL.
Implements the graph neural network that processes discourse structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .discourse_structures import DiscourseGraph, BiaffineAttention


class EGCNLayer(nn.Module):
    """Single layer of Edge-Aware Graph Convolutional Network."""
    
    def __init__(self, input_dim: int, output_dim: int, edge_label_dim: int = 10,
                 activation: str = "relu", dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_label_dim = edge_label_dim
        
        # Edge label embedding
        self.edge_label_embedding = nn.Parameter(torch.randn(edge_label_dim))
        
        # Linear transformation for concatenated features
        # [node_i, node_j, edge_label] -> output_dim
        self.linear = nn.Linear(input_dim * 2 + edge_label_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Biaffine attention for computing aggregation weights
        self.biaffine_attn = BiaffineAttention(input_dim + edge_label_dim, input_dim)
        
        # Activation and dropout
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.edge_label_embedding, std=0.1)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EGCN layer.
        
        Args:
            node_features: [num_nodes, input_dim] - node feature matrix
            edge_index: [2, num_edges] - edge indices
            edge_weights: [num_edges] - edge weights
            
        Returns:
            updated_features: [num_nodes, output_dim] - updated node features
        """
        num_nodes = node_features.size(0)
        device = node_features.device
        
        if edge_index.size(1) == 0:
            # No edges, return transformed node features
            dummy_features = torch.cat([
                node_features, 
                node_features, 
                self.edge_label_embedding.unsqueeze(0).expand(num_nodes, -1)
            ], dim=1)
            return self.activation(self.linear(dummy_features) + self.bias)
        
        # Gather source and target node features
        source_idx, target_idx = edge_index[0], edge_index[1]
        source_features = node_features[source_idx]  # [num_edges, input_dim]
        target_features = node_features[target_idx]  # [num_edges, input_dim]
        
        # Expand edge label embedding for all edges
        edge_labels = self.edge_label_embedding.unsqueeze(0).expand(edge_index.size(1), -1)
        
        # Concatenate features: [source, target, edge_label]
        concatenated = torch.cat([source_features, target_features, edge_labels], dim=1)
        
        # Linear transformation
        transformed = self.linear(concatenated) + self.bias  # [num_edges, output_dim]
        
        # Compute aggregation weights using biaffine attention
        source_with_edge = torch.cat([source_features, edge_labels], dim=1)
        target_with_edge = torch.cat([target_features, edge_labels], dim=1)
        
        attention_scores = self.biaffine_attn(source_with_edge, target_with_edge)  # [num_edges]
        
        # Combine with edge weights and normalize
        combined_weights = edge_weights * torch.exp(attention_scores)
        
        # Compute normalization factor for each target node
        normalization = torch.zeros(num_nodes, device=device)
        normalization.scatter_add_(0, target_idx, combined_weights)
        normalization = normalization[target_idx] + 1e-8  # Add small epsilon to avoid division by zero
        
        # Normalize weights
        normalized_weights = combined_weights / normalization
        
        # Weight the transformed features
        weighted_features = transformed * normalized_weights.unsqueeze(1)
        
        # Aggregate features for each node
        aggregated = torch.zeros(num_nodes, self.output_dim, device=device)
        aggregated.scatter_add_(0, target_idx.unsqueeze(1).expand(-1, self.output_dim), weighted_features)
        
        # Apply activation and dropout
        output = self.activation(aggregated)
        output = self.dropout(output)
        
        return output


class EGCN(nn.Module):
    """Multi-layer Edge-Aware Graph Convolutional Network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], edge_label_dim: int = 10,
                 activation: str = "relu", dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = len(hidden_dims)
        self.layers = nn.ModuleList()
        
        # Build layers
        dims = [input_dim] + hidden_dims
        for i in range(self.num_layers):
            layer = EGCNLayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                edge_label_dim=edge_label_dim,
                activation=activation,
                dropout=dropout
            )
            self.layers.append(layer)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all EGCN layers.
        
        Args:
            node_features: [num_nodes, input_dim] - initial node features
            edge_index: [2, num_edges] - edge indices
            edge_weights: [num_edges] - edge weights
            
        Returns:
            final_features: [num_nodes, hidden_dims[-1]] - final node representations
        """
        x = node_features
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights)
        
        return x


class DiscourseStructureEncoder(nn.Module):
    """Encodes multiple discourse structures using separate EGCNs."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        input_dim = config['input_dim']
        hidden_dims = config['hidden_dims']
        edge_label_dim = config.get('edge_label_dim', 10)
        activation = config.get('activation', 'relu')
        dropout = config.get('dropout', 0.2)
        
        # Create separate EGCN for each discourse structure type
        self.structure_types = ['speaker_utterance', 'speaker_mentioning', 
                               'utterance_distance', 'partial_replying']
        
        self.egcns = nn.ModuleDict()
        for structure_type in self.structure_types:
            if config.get(f'enable_{structure_type}', True):
                self.egcns[structure_type] = EGCN(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    edge_label_dim=edge_label_dim,
                    activation=activation,
                    dropout=dropout
                )
        
        # Fusion method for combining multiple structure representations
        self.fusion_method = config.get('fusion_method', 'addition')
        if self.fusion_method == 'attention':
            self.attention_weights = nn.Parameter(torch.ones(len(self.egcns)))
        elif self.fusion_method == 'mlp':
            fusion_input_dim = hidden_dims[-1] * len(self.egcns)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[-1], hidden_dims[-1])
            )
    
    def forward(self, node_features: torch.Tensor, 
                discourse_graphs: Dict[str, DiscourseGraph]) -> torch.Tensor:
        """
        Encode node features using multiple discourse structures.
        
        Args:
            node_features: [num_nodes, input_dim] - initial node features
            discourse_graphs: Dictionary of discourse graphs by type
            
        Returns:
            fused_features: [num_nodes, hidden_dim] - fused representations
        """
        structure_representations = []
        
        # Process each discourse structure
        for structure_type, egcn in self.egcns.items():
            if structure_type in discourse_graphs:
                graph = discourse_graphs[structure_type]
                
                # Handle empty graphs
                if graph.edge_index.size(1) == 0:
                    # Create self-loop edges for all nodes
                    num_nodes = node_features.size(0)
                    edge_index = torch.arange(num_nodes, device=node_features.device).repeat(2, 1)
                    edge_weights = torch.ones(num_nodes, device=node_features.device)
                else:
                    edge_index = graph.edge_index.to(node_features.device)
                    edge_weights = graph.edge_weights.to(node_features.device)
                
                # Encode using EGCN
                representation = egcn(node_features, edge_index, edge_weights)
                structure_representations.append(representation)
            else:
                # If structure is missing, use original node features
                structure_representations.append(node_features)
        
        # Fuse representations
        if len(structure_representations) == 1:
            return structure_representations[0]
        
        if self.fusion_method == 'addition':
            # Element-wise addition as in the paper (Equation 10)
            fused = sum(structure_representations)
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            weights = F.softmax(self.attention_weights, dim=0)
            fused = sum(w * rep for w, rep in zip(weights, structure_representations))
        elif self.fusion_method == 'mlp':
            # MLP-based fusion
            concatenated = torch.cat(structure_representations, dim=1)
            fused = self.fusion_mlp(concatenated)
        else:
            # Default to addition
            fused = sum(structure_representations)
        
        return fused


class DropConnectLSTM(nn.Module):
    """BiLSTM with DropConnect regularization as described in the paper."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 dropout_feedforward: float = 0.2, dropout_recurrent: float = 0.2,
                 bidirectional: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_feedforward = dropout_feedforward
        self.dropout_recurrent = dropout_recurrent
        self.bidirectional = bidirectional
        
        # Create LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_feedforward if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # DropConnect masks for recurrent connections
        self.register_buffer('recurrent_masks', None)
        
    def _apply_dropconnect(self):
        """Apply DropConnect to recurrent weight matrices during training."""
        if self.training and self.dropout_recurrent > 0:
            for name, param in self.lstm.named_parameters():
                if 'weight_hh' in name:  # Recurrent weights
                    mask = torch.bernoulli(torch.ones_like(param) * (1 - self.dropout_recurrent))
                    param.data *= mask
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through DropConnect LSTM.
        
        Args:
            x: [batch_size, seq_len, input_size] - input sequences
            hidden: Optional initial hidden states
            
        Returns:
            output: [batch_size, seq_len, hidden_size * num_directions] - output sequences
            hidden: Final hidden states
        """
        # Apply DropConnect to recurrent weights
        self._apply_dropconnect()
        
        # LSTM forward pass
        output, hidden = self.lstm(x, hidden)
        
        return output, hidden
