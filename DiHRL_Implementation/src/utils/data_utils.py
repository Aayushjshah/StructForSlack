"""
Data processing utilities for DiHRL dialogue disentanglement.
Based on "Revisiting Conversation Discourse for Dialogue Disentanglement" paper.
"""

import os
import re
import json
import pickle
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import torch
import numpy as np
from collections import defaultdict


@dataclass
class Utterance:
    """Represents a single utterance in a dialogue."""
    id: int
    speaker: str
    text: str
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        # Clean the text
        self.text = self.clean_text(self.text)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean utterance text."""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


@dataclass
class Dialogue:
    """Represents a complete dialogue with utterances and reply-to relations."""
    id: str
    utterances: List[Utterance]
    reply_to: Dict[int, int]  # utterance_id -> parent_utterance_id
    sessions: Optional[List[List[int]]] = None  # List of session clusters
    
    def __post_init__(self):
        # Sort utterances by ID to ensure chronological order
        self.utterances.sort(key=lambda x: x.id)
        
        # Build sessions from reply_to relations if not provided
        if self.sessions is None:
            self.sessions = self.build_sessions()
    
    def build_sessions(self) -> List[List[int]]:
        """Build session clusters from reply-to relations."""
        sessions = []
        visited = set()
        
        for utterance in self.utterances:
            if utterance.id in visited:
                continue
                
            # Find the root of this thread
            session = self.get_session_for_utterance(utterance.id)
            if session:
                sessions.append(session)
                visited.update(session)
        
        return sessions
    
    def get_session_for_utterance(self, utterance_id: int) -> List[int]:
        """Get the complete session/thread for a given utterance."""
        # Find root
        root = utterance_id
        while root in self.reply_to and self.reply_to[root] != root:
            root = self.reply_to[root]
        
        # Collect all utterances in this thread
        session = []
        self._collect_thread_utterances(root, session)
        return sorted(session)
    
    def _collect_thread_utterances(self, utterance_id: int, session: List[int]):
        """Recursively collect all utterances in a thread."""
        session.append(utterance_id)
        # Find children
        for uid, parent in self.reply_to.items():
            if parent == utterance_id and uid != utterance_id:
                self._collect_thread_utterances(uid, session)
    
    def get_speaker_utterances(self, speaker: str) -> List[int]:
        """Get all utterance IDs for a specific speaker."""
        return [u.id for u in self.utterances if u.speaker == speaker]
    
    def get_mentioned_speakers(self, utterance_id: int) -> Set[str]:
        """Extract mentioned speakers from an utterance."""
        utterance = next((u for u in self.utterances if u.id == utterance_id), None)
        if not utterance:
            return set()
        
        # Simple heuristic: look for speaker names in the text
        mentioned = set()
        all_speakers = {u.speaker for u in self.utterances}
        
        for speaker in all_speakers:
            if speaker != utterance.speaker and speaker.lower() in utterance.text.lower():
                mentioned.add(speaker)
        
        return mentioned


class IRCDataLoader:
    """Loads and processes IRC dialogue disentanglement data."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_dialogues(self, split: str) -> List[Dialogue]:
        """Load dialogues for a specific split (train/dev/test)."""
        dialogues = []
        
        # Load utterances
        utterances_file = os.path.join(self.data_path, split, "utterances.txt")
        dialogue_utterances = self._load_utterances(utterances_file)
        
        # Load reply-to relations
        graphs_file = os.path.join(self.data_path, f"gold.{split}.graphs.txt")
        dialogue_graphs = self._load_graphs(graphs_file)
        
        # Load cluster information
        clusters_file = os.path.join(self.data_path, f"gold.{split}.clusters.txt")
        dialogue_clusters = self._load_clusters(clusters_file)
        
        # Combine into Dialogue objects
        for dialogue_id in dialogue_utterances:
            utterances = dialogue_utterances[dialogue_id]
            reply_to = dialogue_graphs.get(dialogue_id, {})
            sessions = dialogue_clusters.get(dialogue_id, None)
            
            dialogue = Dialogue(
                id=dialogue_id,
                utterances=utterances,
                reply_to=reply_to,
                sessions=sessions
            )
            dialogues.append(dialogue)
        
        return dialogues
    
    def _load_utterances(self, file_path: str) -> Dict[str, List[Utterance]]:
        """Load utterances from file."""
        dialogue_utterances = defaultdict(list)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Looking for alternative format...")
            # Try to find files in the directory
            split_dir = os.path.dirname(file_path)
            for file in os.listdir(split_dir):
                if file.endswith('.txt'):
                    print(f"Found file: {file}")
            return dialogue_utterances
        
        with open(file_path, 'r', encoding='utf-8') as f:
            current_dialogue = None
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('# Dialogue:'):
                    current_dialogue = line.split(':', 1)[1].strip()
                elif current_dialogue and '\t' in line:
                    parts = line.split('\t', 2)
                    if len(parts) >= 3:
                        utterance_id = int(parts[0])
                        speaker = parts[1]
                        text = parts[2]
                        
                        utterance = Utterance(
                            id=utterance_id,
                            speaker=speaker,
                            text=text
                        )
                        dialogue_utterances[current_dialogue].append(utterance)
        
        return dict(dialogue_utterances)
    
    def _load_graphs(self, file_path: str) -> Dict[str, Dict[int, int]]:
        """Load reply-to graphs from file."""
        dialogue_graphs = {}
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return dialogue_graphs
        
        with open(file_path, 'r', encoding='utf-8') as f:
            current_dialogue = None
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('# Dialogue:'):
                    current_dialogue = line.split(':', 1)[1].strip()
                    dialogue_graphs[current_dialogue] = {}
                elif current_dialogue and '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        child = int(parts[0].strip())
                        parent = int(parts[1].strip())
                        dialogue_graphs[current_dialogue][child] = parent
        
        return dialogue_graphs
    
    def _load_clusters(self, file_path: str) -> Dict[str, List[List[int]]]:
        """Load session clusters from file."""
        dialogue_clusters = {}
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            return dialogue_clusters
        
        with open(file_path, 'r', encoding='utf-8') as f:
            current_dialogue = None
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('# Dialogue:'):
                    current_dialogue = line.split(':', 1)[1].strip()
                    dialogue_clusters[current_dialogue] = []
                elif current_dialogue:
                    # Parse cluster
                    cluster_str = line.strip('[]')
                    if cluster_str:
                        cluster = [int(x.strip()) for x in cluster_str.split(',') if x.strip()]
                        dialogue_clusters[current_dialogue].append(cluster)
        
        return dialogue_clusters


def collate_dialogues(dialogues: List[Dialogue]) -> Dict:
    """Collate function for DataLoader."""
    return {
        'dialogues': dialogues,
        'num_dialogues': len(dialogues)
    }


def save_processed_data(data: List[Dialogue], file_path: str):
    """Save processed dialogues to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_processed_data(file_path: str) -> List[Dialogue]:
    """Load processed dialogues from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
