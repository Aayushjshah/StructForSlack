"""
Easy-First Decoding Algorithm for DiHRL.
Implements the non-sequential decoding strategy described in the paper.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict

from .dihrl_model import DiHRLModel
from ..utils.data_utils import Dialogue


class EasyFirstDecoder:
    """
    Easy-First Decoding for dialogue disentanglement.
    Implements Algorithm 1 from the paper.
    """
    
    def __init__(self, model: DiHRLModel, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.context_window_size = config['data']['context_window_size']
        self.confidence_threshold = config['decoding'].get('confidence_threshold', 0.5)
        
    def decode_dialogue(self, dialogue: Dialogue) -> Dict[int, int]:
        """
        Decode reply-to relations for an entire dialogue using easy-first strategy.
        
        Args:
            dialogue: Complete dialogue object
            
        Returns:
            reply_to_relations: Dictionary mapping utterance_id -> parent_utterance_id
        """
        self.model.eval()
        
        # Initialize structures
        U = set(u.id for u in dialogue.utterances)  # Current dialogue queue
        Q = set(u.id for u in dialogue.utterances)  # Memory of candidate parents
        partial_replying_edges = {}  # Established reply relations with confidence scores
        reply_relations = {}  # Final reply-to mapping
        
        # Sliding window tracking
        processed_utterances = set()
        
        while U:
            # Step 1: Compute pairwise scoring matrix for current state
            scores_matrix = self._compute_pairwise_scores(dialogue, U, Q, partial_replying_edges)
            
            if not scores_matrix:
                # No valid pairs found, process remaining utterances sequentially
                for utterance_id in sorted(U):
                    reply_relations[utterance_id] = utterance_id  # Self-reference
                break
            
            # Step 2: Find the highest-scored parent-child pair
            best_child, best_parent, best_score = self._find_best_pair(scores_matrix)
            
            if best_score < self.confidence_threshold:
                # Low confidence, fall back to sequential processing
                remaining_utterances = sorted(U)
                for utterance_id in remaining_utterances:
                    if utterance_id not in reply_relations:
                        reply_relations[utterance_id] = utterance_id
                break
            
            # Step 3: Establish the reply relation
            reply_relations[best_child] = best_parent
            partial_replying_edges[(best_child, best_parent)] = best_score
            
            # Step 4: Update structures
            U.discard(best_child)  # Remove child from queue
            processed_utterances.add(best_child)
            
            # Step 5: Sliding window update
            # Add next utterance in window if available
            next_utterance_id = best_child + self.context_window_size + 1
            if next_utterance_id <= max(u.id for u in dialogue.utterances):
                # Find the actual next utterance
                for utterance in dialogue.utterances:
                    if (utterance.id > best_child and 
                        utterance.id not in processed_utterances and 
                        utterance.id not in U):
                        U.add(utterance.id)
                        Q.add(utterance.id)
                        break
        
        # Ensure all utterances have parents
        for utterance in dialogue.utterances:
            if utterance.id not in reply_relations:
                reply_relations[utterance.id] = utterance.id
        
        return reply_relations
    
    def _compute_pairwise_scores(self, dialogue: Dialogue, U: Set[int], Q: Set[int],
                                partial_replying_edges: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        """
        Compute pairwise scores for all valid (child, parent) pairs.
        
        Args:
            dialogue: Complete dialogue object
            U: Set of current utterance IDs
            Q: Set of candidate parent IDs
            partial_replying_edges: Current partial replying structure
            
        Returns:
            scores_matrix: Dictionary mapping (child_id, parent_id) -> score
        """
        scores_matrix = {}
        
        with torch.no_grad():
            for child_id in U:
                # Find valid candidate parents for this child
                candidate_parents = []
                for parent_id in Q:
                    # Parent must be chronologically before or equal to child
                    if parent_id <= child_id:
                        # Within context window
                        if child_id - parent_id <= self.context_window_size:
                            candidate_parents.append(parent_id)
                
                if not candidate_parents:
                    continue
                
                # Get model predictions for this child
                try:
                    scores = self.model.forward(
                        dialogue=dialogue,
                        current_utterance_id=child_id,
                        candidate_utterance_ids=candidate_parents,
                        tokenizer=self.tokenizer,
                        partial_replying_edges=partial_replying_edges
                    )
                    
                    # Convert to probabilities
                    probabilities = F.softmax(scores, dim=0)
                    
                    # Store scores for all candidate pairs
                    for i, parent_id in enumerate(candidate_parents):
                        scores_matrix[(child_id, parent_id)] = probabilities[i].item()
                        
                except Exception as e:
                    print(f"Error processing utterance {child_id}: {e}")
                    continue
        
        return scores_matrix
    
    def _find_best_pair(self, scores_matrix: Dict[Tuple[int, int], float]) -> Tuple[int, int, float]:
        """
        Find the (child, parent) pair with the highest confidence score.
        
        Args:
            scores_matrix: Dictionary mapping (child_id, parent_id) -> score
            
        Returns:
            best_child: Child utterance ID
            best_parent: Parent utterance ID  
            best_score: Confidence score
        """
        if not scores_matrix:
            return None, None, 0.0
        
        # Find pair with maximum score
        best_pair = max(scores_matrix.items(), key=lambda x: x[1])
        (best_child, best_parent), best_score = best_pair
        
        return best_child, best_parent, best_score
    
    def decode_dialogue_sequential(self, dialogue: Dialogue) -> Dict[int, int]:
        """
        Baseline sequential decoding for comparison.
        
        Args:
            dialogue: Complete dialogue object
            
        Returns:
            reply_to_relations: Dictionary mapping utterance_id -> parent_utterance_id
        """
        self.model.eval()
        reply_relations = {}
        partial_replying_edges = {}
        
        # Process utterances in chronological order
        sorted_utterances = sorted(dialogue.utterances, key=lambda u: u.id)
        
        with torch.no_grad():
            for utterance in sorted_utterances:
                parent_id, _ = self.model.predict_parent(
                    dialogue=dialogue,
                    current_utterance_id=utterance.id,
                    tokenizer=self.tokenizer,
                    partial_replying_edges=partial_replying_edges
                )
                
                reply_relations[utterance.id] = parent_id
                
                # Update partial replying structure
                if parent_id != utterance.id:
                    partial_replying_edges[(utterance.id, parent_id)] = 1.0
        
        return reply_relations


class DecodingEvaluator:
    """Evaluates the effectiveness of different decoding strategies."""
    
    def __init__(self, model: DiHRLModel, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.easy_first_decoder = EasyFirstDecoder(model, tokenizer, config)
    
    def compare_decoding_strategies(self, dialogue: Dialogue) -> Dict[str, Dict[int, int]]:
        """
        Compare easy-first vs sequential decoding on a dialogue.
        
        Args:
            dialogue: Complete dialogue object
            
        Returns:
            results: Dictionary with results from both strategies
        """
        results = {}
        
        # Easy-first decoding
        try:
            easy_first_results = self.easy_first_decoder.decode_dialogue(dialogue)
            results['easy_first'] = easy_first_results
        except Exception as e:
            print(f"Easy-first decoding failed: {e}")
            results['easy_first'] = {}
        
        # Sequential decoding
        try:
            sequential_results = self.easy_first_decoder.decode_dialogue_sequential(dialogue)
            results['sequential'] = sequential_results
        except Exception as e:
            print(f"Sequential decoding failed: {e}")
            results['sequential'] = {}
        
        return results
    
    def analyze_decoding_confidence(self, dialogue: Dialogue) -> Dict[str, List[float]]:
        """
        Analyze the confidence trajectories of different decoding strategies.
        
        Args:
            dialogue: Complete dialogue object
            
        Returns:
            confidence_trajectories: Dictionary mapping strategy -> confidence scores over time
        """
        self.model.eval()
        
        trajectories = {
            'easy_first': [],
            'sequential': []
        }
        
        # Easy-first trajectory
        U = set(u.id for u in dialogue.utterances)
        Q = set(u.id for u in dialogue.utterances)
        partial_replying_edges = {}
        
        with torch.no_grad():
            while U:
                scores_matrix = self.easy_first_decoder._compute_pairwise_scores(
                    dialogue, U, Q, partial_replying_edges
                )
                
                if not scores_matrix:
                    break
                
                _, _, best_score = self.easy_first_decoder._find_best_pair(scores_matrix)
                trajectories['easy_first'].append(best_score)
                
                # Simulate one step of easy-first decoding
                best_child, best_parent, _ = self.easy_first_decoder._find_best_pair(scores_matrix)
                U.discard(best_child)
                partial_replying_edges[(best_child, best_parent)] = best_score
        
        # Sequential trajectory
        sorted_utterances = sorted(dialogue.utterances, key=lambda u: u.id)
        partial_replying_edges = {}
        
        with torch.no_grad():
            for utterance in sorted_utterances:
                parent_id, probabilities = self.model.predict_parent(
                    dialogue=dialogue,
                    current_utterance_id=utterance.id,
                    tokenizer=self.tokenizer,
                    partial_replying_edges=partial_replying_edges
                )
                
                # Record maximum probability as confidence
                max_confidence = torch.max(probabilities).item()
                trajectories['sequential'].append(max_confidence)
                
                # Update partial structure
                if parent_id != utterance.id:
                    partial_replying_edges[(utterance.id, parent_id)] = 1.0
        
        return trajectories
