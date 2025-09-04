"""
Evaluation metrics for dialogue disentanglement.
Implements all metrics mentioned in the DiHRL paper.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import math


class DialogueDisentanglementMetrics:
    """Comprehensive evaluation metrics for dialogue disentanglement."""
    
    def __init__(self):
        pass
    
    def evaluate_all(self, predicted_clusters: List[List[int]], 
                    gold_clusters: List[List[int]],
                    predicted_links: Dict[int, int],
                    gold_links: Dict[int, int]) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predicted_clusters: List of predicted session clusters
            gold_clusters: List of gold session clusters  
            predicted_links: Predicted reply-to relations {child: parent}
            gold_links: Gold reply-to relations {child: parent}
            
        Returns:
            metrics: Dictionary of all computed metrics
        """
        results = {}
        
        # Cluster-level metrics
        cluster_metrics = self.compute_cluster_metrics(predicted_clusters, gold_clusters)
        results.update(cluster_metrics)
        
        # Pairwise link metrics
        link_metrics = self.compute_link_metrics(predicted_links, gold_links)
        results.update(link_metrics)
        
        return results
    
    def compute_cluster_metrics(self, predicted_clusters: List[List[int]], 
                               gold_clusters: List[List[int]]) -> Dict[str, float]:
        """
        Compute cluster-level evaluation metrics.
        
        Args:
            predicted_clusters: List of predicted session clusters
            gold_clusters: List of gold session clusters
            
        Returns:
            metrics: Dictionary of cluster-level metrics
        """
        results = {}
        
        # Convert clusters to label arrays for sklearn metrics
        pred_labels, gold_labels = self._clusters_to_labels(predicted_clusters, gold_clusters)
        
        if len(pred_labels) == 0 or len(gold_labels) == 0:
            return {metric: 0.0 for metric in [
                'variation_information', 'adjusted_rand_index', 'one_to_one',
                'normalized_mutual_information', 'local_k', 'shen_f1',
                'cluster_precision', 'cluster_recall', 'cluster_f1'
            ]}
        
        # Variation of Information (VI)
        results['variation_information'] = 1 - self._compute_variation_information(pred_labels, gold_labels)
        
        # Adjusted Rand Index (ARI)
        results['adjusted_rand_index'] = adjusted_rand_score(gold_labels, pred_labels)
        
        # One-to-One (1-1)
        results['one_to_one'] = self._compute_one_to_one(predicted_clusters, gold_clusters)
        
        # Normalized Mutual Information (NMI)
        results['normalized_mutual_information'] = normalized_mutual_info_score(
            gold_labels, pred_labels, average_method='arithmetic'
        )
        
        # Local_k metric (k=3 as specified in paper)
        results['local_k'] = self._compute_local_k(pred_labels, gold_labels, k=3)
        
        # Shen-F1
        results['shen_f1'] = self._compute_shen_f1(predicted_clusters, gold_clusters)
        
        # Cluster Exact Match (Precision, Recall, F1)
        cluster_p, cluster_r, cluster_f1 = self._compute_cluster_exact_match(
            predicted_clusters, gold_clusters
        )
        results['cluster_precision'] = cluster_p
        results['cluster_recall'] = cluster_r
        results['cluster_f1'] = cluster_f1
        
        return results
    
    def compute_link_metrics(self, predicted_links: Dict[int, int],
                            gold_links: Dict[int, int]) -> Dict[str, float]:
        """
        Compute pairwise link evaluation metrics.
        
        Args:
            predicted_links: Predicted reply-to relations {child: parent}
            gold_links: Gold reply-to relations {child: parent}
            
        Returns:
            metrics: Dictionary of link-level metrics
        """
        # Convert to sets of (child, parent) pairs
        pred_pairs = set((child, parent) for child, parent in predicted_links.items())
        gold_pairs = set((child, parent) for child, parent in gold_links.items())
        
        # Compute precision, recall, F1
        if len(pred_pairs) == 0:
            precision = 0.0
        else:
            precision = len(pred_pairs & gold_pairs) / len(pred_pairs)
        
        if len(gold_pairs) == 0:
            recall = 0.0
        else:
            recall = len(pred_pairs & gold_pairs) / len(gold_pairs)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'link_precision': precision,
            'link_recall': recall,
            'link_f1': f1
        }
    
    def compute_partial_ari(self, predicted_clusters: List[List[int]],
                           gold_clusters: List[List[int]],
                           size_range: Tuple[int, int]) -> float:
        """
        Compute Partial-ARI for clusters within a specific size range.
        
        Args:
            predicted_clusters: List of predicted session clusters
            gold_clusters: List of gold session clusters
            size_range: (min_size, max_size) for filtering clusters
            
        Returns:
            partial_ari: Partial ARI score
        """
        min_size, max_size = size_range
        
        # Filter gold clusters by size
        filtered_gold = [cluster for cluster in gold_clusters 
                        if min_size <= len(cluster) <= max_size]
        
        if not filtered_gold:
            return 0.0
        
        # Find corresponding predicted clusters
        filtered_pred = []
        gold_utterances = set()
        for cluster in filtered_gold:
            gold_utterances.update(cluster)
        
        for pred_cluster in predicted_clusters:
            if any(utterance in gold_utterances for utterance in pred_cluster):
                # This predicted cluster overlaps with our filtered gold clusters
                filtered_pred.append(pred_cluster)
        
        if not filtered_pred:
            return 0.0
        
        # Compute ARI on filtered clusters
        pred_labels, gold_labels = self._clusters_to_labels(filtered_pred, filtered_gold)
        
        if len(pred_labels) == 0 or len(gold_labels) == 0:
            return 0.0
        
        return adjusted_rand_score(gold_labels, pred_labels)
    
    def _clusters_to_labels(self, predicted_clusters: List[List[int]],
                           gold_clusters: List[List[int]]) -> Tuple[List[int], List[int]]:
        """Convert cluster lists to label arrays for sklearn metrics."""
        # Get all utterance IDs
        all_utterances = set()
        for cluster in predicted_clusters + gold_clusters:
            all_utterances.update(cluster)
        
        all_utterances = sorted(all_utterances)
        
        # Create label arrays
        pred_labels = [-1] * len(all_utterances)
        gold_labels = [-1] * len(all_utterances)
        
        utterance_to_idx = {uid: i for i, uid in enumerate(all_utterances)}
        
        # Assign predicted labels
        for cluster_id, cluster in enumerate(predicted_clusters):
            for utterance_id in cluster:
                if utterance_id in utterance_to_idx:
                    pred_labels[utterance_to_idx[utterance_id]] = cluster_id
        
        # Assign gold labels
        for cluster_id, cluster in enumerate(gold_clusters):
            for utterance_id in cluster:
                if utterance_id in utterance_to_idx:
                    gold_labels[utterance_to_idx[utterance_id]] = cluster_id
        
        return pred_labels, gold_labels
    
    def _compute_variation_information(self, pred_labels: List[int], 
                                     gold_labels: List[int]) -> float:
        """Compute Variation of Information between two clusterings."""
        if len(pred_labels) != len(gold_labels):
            return float('inf')
        
        n = len(pred_labels)
        if n == 0:
            return 0.0
        
        # Compute contingency matrix
        contingency = contingency_matrix(gold_labels, pred_labels)
        
        # Compute marginal probabilities
        pred_probs = contingency.sum(axis=0) / n
        gold_probs = contingency.sum(axis=1) / n
        joint_probs = contingency / n
        
        # Compute entropies
        def entropy(probs):
            return -sum(p * math.log2(p) for p in probs if p > 0)
        
        h_pred = entropy(pred_probs)
        h_gold = entropy(gold_probs)
        
        # Compute joint entropy
        h_joint = entropy(joint_probs.flatten())
        
        # VI = H(X,Y) - I(X;Y) = 2*H(X,Y) - H(X) - H(Y)
        vi = 2 * h_joint - h_pred - h_gold
        
        return vi
    
    def _compute_one_to_one(self, predicted_clusters: List[List[int]],
                           gold_clusters: List[List[int]]) -> float:
        """Compute One-to-One matching score."""
        if not predicted_clusters or not gold_clusters:
            return 0.0
        
        # Create bipartite matching between predicted and gold clusters
        # Use Jaccard similarity as matching criterion
        max_similarity = 0.0
        used_gold = set()
        total_matched = 0
        
        for pred_cluster in predicted_clusters:
            best_match_idx = -1
            best_similarity = 0.0
            
            for i, gold_cluster in enumerate(gold_clusters):
                if i in used_gold:
                    continue
                
                # Compute Jaccard similarity
                pred_set = set(pred_cluster)
                gold_set = set(gold_cluster)
                
                if len(pred_set | gold_set) == 0:
                    similarity = 0.0
                else:
                    similarity = len(pred_set & gold_set) / len(pred_set | gold_set)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            if best_match_idx >= 0:
                used_gold.add(best_match_idx)
                max_similarity += best_similarity
                total_matched += 1
        
        return max_similarity / max(len(predicted_clusters), len(gold_clusters))
    
    def _compute_local_k(self, pred_labels: List[int], gold_labels: List[int], k: int = 3) -> float:
        """Compute Local_k accuracy."""
        if len(pred_labels) <= k or len(gold_labels) <= k:
            return 0.0
        
        total_comparisons = 0
        correct_comparisons = 0
        
        for i in range(k, len(pred_labels)):
            for j in range(1, k + 1):
                if i - j >= 0:
                    # Check if utterances i and i-j are in same cluster
                    pred_same = (pred_labels[i] == pred_labels[i - j])
                    gold_same = (gold_labels[i] == gold_labels[i - j])
                    
                    total_comparisons += 1
                    if pred_same == gold_same:
                        correct_comparisons += 1
        
        return correct_comparisons / total_comparisons if total_comparisons > 0 else 0.0
    
    def _compute_shen_f1(self, predicted_clusters: List[List[int]],
                        gold_clusters: List[List[int]]) -> float:
        """Compute Shen-F1 score."""
        if not predicted_clusters or not gold_clusters:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for gold_cluster in gold_clusters:
            gold_set = set(gold_cluster)
            cluster_size = len(gold_set)
            
            if cluster_size == 0:
                continue
            
            # Find best matching predicted cluster
            best_f1 = 0.0
            
            for pred_cluster in predicted_clusters:
                pred_set = set(pred_cluster)
                
                intersection = len(gold_set & pred_set)
                
                if len(pred_set) == 0:
                    precision = 0.0
                else:
                    precision = intersection / len(pred_set)
                
                recall = intersection / len(gold_set)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                
                best_f1 = max(best_f1, f1)
            
            total_score += cluster_size * best_f1
            total_weight += cluster_size
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _compute_cluster_exact_match(self, predicted_clusters: List[List[int]],
                                   gold_clusters: List[List[int]]) -> Tuple[float, float, float]:
        """Compute cluster-level exact match precision, recall, and F1."""
        if not predicted_clusters and not gold_clusters:
            return 1.0, 1.0, 1.0
        
        if not predicted_clusters:
            return 0.0, 0.0, 0.0
        
        if not gold_clusters:
            return 0.0, 0.0, 0.0
        
        # Convert to sets for exact matching
        pred_cluster_sets = [set(cluster) for cluster in predicted_clusters]
        gold_cluster_sets = [set(cluster) for cluster in gold_clusters]
        
        # Count exact matches
        matched_pred = 0
        for pred_set in pred_cluster_sets:
            if pred_set in gold_cluster_sets:
                matched_pred += 1
        
        matched_gold = 0
        for gold_set in gold_cluster_sets:
            if gold_set in pred_cluster_sets:
                matched_gold += 1
        
        # Compute precision, recall, F1
        precision = matched_pred / len(pred_cluster_sets)
        recall = matched_gold / len(gold_cluster_sets)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return precision, recall, f1


def links_to_clusters(links: Dict[int, int]) -> List[List[int]]:
    """
    Convert reply-to links to session clusters.
    
    Args:
        links: Dictionary {child_id: parent_id}
        
    Returns:
        clusters: List of session clusters
    """
    # Build threads by following reply chains
    visited = set()
    clusters = []
    
    for utterance_id in links.keys():
        if utterance_id in visited:
            continue
        
        # Find root of this thread
        root = utterance_id
        path = []
        
        while root in links and links[root] != root and root not in path:
            path.append(root)
            root = links[root]
        
        # Collect all utterances in this thread
        thread = set([root])
        stack = [root]
        
        while stack:
            current = stack.pop()
            for child, parent in links.items():
                if parent == current and child not in thread:
                    thread.add(child)
                    stack.append(child)
        
        if thread:
            clusters.append(sorted(thread))
            visited.update(thread)
    
    return clusters


def clusters_to_links(clusters: List[List[int]]) -> Dict[int, int]:
    """
    Convert session clusters to reply-to links (approximate).
    
    Args:
        clusters: List of session clusters
        
    Returns:
        links: Dictionary {child_id: parent_id}
    """
    links = {}
    
    for cluster in clusters:
        if len(cluster) <= 1:
            # Single utterance cluster - self-reference
            if cluster:
                links[cluster[0]] = cluster[0]
        else:
            # Multi-utterance cluster - create linear chain
            sorted_cluster = sorted(cluster)
            for i, utterance_id in enumerate(sorted_cluster):
                if i == 0:
                    # Root utterance - self-reference
                    links[utterance_id] = utterance_id
                else:
                    # Link to previous utterance
                    links[utterance_id] = sorted_cluster[i - 1]
    
    return links
