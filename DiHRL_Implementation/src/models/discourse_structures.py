"""
Discourse structure construction for DiHRL.
Implements the four types of discourse graphs from the paper:
1. Speaker-Utterance Structure (GS)
2. Speaker-Mentioning Structure (GM)
3. Utterance-Distance Structure (GD)
4. Partial-Replying Structure (GR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import math

from ..utils.data_utils import Dialogue, Utterance


@dataclass
class DiscourseGraph:
    """Represents a discourse graph with nodes and edge weights."""
    edge_index: torch.Tensor  # [2, num_edges]
    edge_weights: torch.Tensor  # [num_edges]
    edge_type: str
    num_nodes: int


class DiscourseStructureBuilder:
    """Builds discourse structure graphs from dialogue data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gaussian_mu = config.get('gaussian_mu', 0)
        self.gaussian_sigma = config.get('gaussian_sigma', 0.7071067811865475)
        self.scale_factor_distance = config.get('scale_factor_distance', 2.5066282746310002)
        
    def build_all_structures(self, dialogue: Dialogue, context_window: Tuple[int, int],
                           partial_replying_edges: Optional[Dict[Tuple[int, int], float]] = None) -> Dict[str, DiscourseGraph]:
        """Build all four types of discourse structures for a dialogue."""
        start_idx, end_idx = context_window
        utterances = [u for u in dialogue.utterances if start_idx <= u.id <= end_idx]
        
        structures = {}
        
        # Build Speaker-Utterance Structure (GS)
        if self.config.get('speaker_utterance', {}).get('enabled', True):
            structures['speaker_utterance'] = self.build_speaker_utterance_structure(utterances)
        
        # Build Speaker-Mentioning Structure (GM)
        if self.config.get('speaker_mentioning', {}).get('enabled', True):
            structures['speaker_mentioning'] = self.build_speaker_mentioning_structure(dialogue, utterances)
        
        # Build Utterance-Distance Structure (GD)
        if self.config.get('utterance_distance', {}).get('enabled', True):
            structures['utterance_distance'] = self.build_utterance_distance_structure(utterances)
        
        # Build Partial-Replying Structure (GR)
        if self.config.get('partial_replying', {}).get('enabled', True):
            structures['partial_replying'] = self.build_partial_replying_structure(
                utterances, partial_replying_edges or {}
            )
        
        return structures
    
    def build_speaker_utterance_structure(self, utterances: List[Utterance]) -> DiscourseGraph:
        """Build speaker-utterance structure connecting utterances from same speaker."""
        edge_list = []
        edge_weights = []
        
        # Group utterances by speaker
        speaker_utterances = {}
        for utterance in utterances:
            if utterance.speaker not in speaker_utterances:
                speaker_utterances[utterance.speaker] = []
            speaker_utterances[utterance.speaker].append(utterance.id)
        
        # Create edges between utterances from the same speaker
        for speaker, utterance_ids in speaker_utterances.items():
            for i, uid1 in enumerate(utterance_ids):
                for j, uid2 in enumerate(utterance_ids):
                    if i != j:  # Don't connect utterance to itself
                        edge_list.append([uid1, uid2])
                        edge_weights.append(1.0)
        
        # Add self-loops
        for utterance in utterances:
            edge_list.append([utterance.id, utterance.id])
            edge_weights.append(1.0)
        
        if not edge_list:
            # Create empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return DiscourseGraph(
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_type='speaker_utterance',
            num_nodes=len(utterances)
        )
    
    def build_speaker_mentioning_structure(self, dialogue: Dialogue, utterances: List[Utterance]) -> DiscourseGraph:
        """Build speaker-mentioning structure connecting mentioning utterances to mentioned speakers."""
        edge_list = []
        edge_weights = []
        
        # Create mapping from utterance ID to index in context window
        id_to_idx = {u.id: i for i, u in enumerate(utterances)}
        
        for utterance in utterances:
            mentioned_speakers = dialogue.get_mentioned_speakers(utterance.id)
            
            for mentioned_speaker in mentioned_speakers:
                # Find utterances by the mentioned speaker in the context window
                for other_utterance in utterances:
                    if other_utterance.speaker == mentioned_speaker:
                        edge_list.append([utterance.id, other_utterance.id])
                        edge_weights.append(1.0)
        
        # Add self-loops
        for utterance in utterances:
            edge_list.append([utterance.id, utterance.id])
            edge_weights.append(1.0)
        
        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return DiscourseGraph(
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_type='speaker_mentioning',
            num_nodes=len(utterances)
        )
    
    def build_utterance_distance_structure(self, utterances: List[Utterance]) -> DiscourseGraph:
        """Build utterance-distance structure with Gaussian-weighted edges."""
        edge_list = []
        edge_weights = []
        
        # Sort utterances by ID to ensure correct distance calculation
        sorted_utterances = sorted(utterances, key=lambda x: x.id)
        
        for i, utterance_i in enumerate(sorted_utterances):
            for j, utterance_j in enumerate(sorted_utterances):
                if i != j:
                    # Calculate distance-based weight using Gaussian distribution
                    distance = abs(utterance_i.id - utterance_j.id)
                    
                    # Apply Gaussian weighting: f(d) = exp(-π*d²)
                    gaussian_weight = math.exp(-math.pi * distance * distance)
                    
                    edge_list.append([utterance_i.id, utterance_j.id])
                    edge_weights.append(gaussian_weight)
        
        # Add self-loops
        for utterance in utterances:
            edge_list.append([utterance.id, utterance.id])
            edge_weights.append(1.0)
        
        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return DiscourseGraph(
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_type='utterance_distance',
            num_nodes=len(utterances)
        )
    
    def build_partial_replying_structure(self, utterances: List[Utterance], 
                                       partial_edges: Dict[Tuple[int, int], float]) -> DiscourseGraph:
        """Build partial-replying structure from established reply relations."""
        edge_list = []
        edge_weights = []
        
        utterance_ids = {u.id for u in utterances}
        
        # Add edges from partial replying relations
        for (child_id, parent_id), weight in partial_edges.items():
            if child_id in utterance_ids and parent_id in utterance_ids:
                # Bidirectional edges for replying relations
                edge_list.append([child_id, parent_id])
                edge_weights.append(weight)
                edge_list.append([parent_id, child_id])
                edge_weights.append(weight)
        
        # Add self-loops
        for utterance in utterances:
            edge_list.append([utterance.id, utterance.id])
            edge_weights.append(1.0)
        
        if not edge_list:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return DiscourseGraph(
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_type='partial_replying',
            num_nodes=len(utterances)
        )


class BiaffineAttention(nn.Module):
    """Biaffine attention mechanism for computing edge weights."""
    
    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Biaffine transformation matrix
        self.W_biaf = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        
        # Linear transformations
        self.linear_i = nn.Linear(input_dim, hidden_dim)
        self.linear_j = nn.Linear(input_dim, hidden_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_biaf)
        nn.init.xavier_uniform_(self.linear_i.weight)
        nn.init.xavier_uniform_(self.linear_j.weight)
        nn.init.zeros_(self.linear_i.bias)
        nn.init.zeros_(self.linear_j.bias)
    
    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        Compute biaffine attention scores.
        
        Args:
            h_i: [batch_size, input_dim] - representations for first set
            h_j: [batch_size, input_dim] - representations for second set
            
        Returns:
            scores: [batch_size] - biaffine attention scores
        """
        # Transform representations
        h_i_proj = self.linear_i(h_i)  # [batch_size, hidden_dim]
        h_j_proj = self.linear_j(h_j)  # [batch_size, hidden_dim]
        
        # Biaffine transformation: h_i^T * W * h_j
        scores = torch.sum(h_i_proj * torch.matmul(h_j_proj, self.W_biaf.t()), dim=-1)
        
        return scores


class UtteranceDistanceComputer:
    """Computes distance-aware weights for utterance pairs."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.gaussian_sigma = config.get('gaussian_sigma', 0.7071067811865475)
        self.scale_factor_distance = config.get('scale_factor_distance', 2.5066282746310002)
    
    def compute_distance_weights(self, utterance_representations: torch.Tensor, 
                               utterance_ids: List[int],
                               biaffine_attn: BiaffineAttention) -> torch.Tensor:
        """
        Compute distance-aware weights as described in Equations 2-3 of the paper.
        
        Args:
            utterance_representations: [num_utterances, hidden_dim]
            utterance_ids: List of utterance IDs in chronological order
            biaffine_attn: Biaffine attention module
            
        Returns:
            distance_weights: [num_utterances, num_utterances] - softmax normalized weights
        """
        num_utterances = len(utterance_ids)
        
        # Compute pairwise biaffine scores
        biaffine_scores = torch.zeros(num_utterances, num_utterances)
        
        for i in range(num_utterances):
            for j in range(num_utterances):
                if i != j:
                    score = biaffine_attn(
                        utterance_representations[i:i+1], 
                        utterance_representations[j:j+1]
                    )
                    biaffine_scores[i, j] = score.item()
        
        # Compute distance-based Gaussian weights
        distance_matrix = torch.zeros(num_utterances, num_utterances)
        
        for i in range(num_utterances):
            for j in range(num_utterances):
                if i != j:
                    distance = abs(utterance_ids[i] - utterance_ids[j])
                    
                    # Gaussian distance weight with scaling
                    distance_term = (distance ** 2) / (2 * math.sqrt(2 * math.pi))
                    
                    # Biaffine term with distance-based scaling
                    biaffine_term = biaffine_scores[i, j] / math.sqrt(distance + 1e-8)
                    
                    # Combine terms as in Equation 3
                    combined_score = -distance_term + biaffine_term
                    distance_matrix[i, j] = combined_score
        
        # Apply softmax normalization row-wise
        distance_weights = F.softmax(distance_matrix, dim=1)
        
        return distance_weights
