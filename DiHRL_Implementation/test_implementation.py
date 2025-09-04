"""
Quick test script to verify DiHRL implementation works correctly.
"""

import os
import sys
import yaml
import torch
from transformers import BertTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.dihrl_model import DiHRLModel, HierarchicalRankingLoss
from src.models.discourse_structures import DiscourseStructureBuilder
from src.models.easy_first_decoder import EasyFirstDecoder
from src.utils.data_utils import Utterance, Dialogue
from src.evaluation.metrics import DialogueDisentanglementMetrics
from src.training.trainer import setup_gpu_training


def create_test_dialogue() -> Dialogue:
    """Create a simple test dialogue for verification."""
    utterances = [
        Utterance(id=0, speaker="alice", text="Hello everyone"),
        Utterance(id=1, speaker="bob", text="Hi alice, how are you?"),
        Utterance(id=2, speaker="alice", text="I'm doing great, thanks!"),
        Utterance(id=3, speaker="charlie", text="Hey alice, what's up?"),
        Utterance(id=4, speaker="alice", text="Not much charlie, just chatting with bob"),
        Utterance(id=5, speaker="bob", text="alice, that's awesome to hear!")
    ]
    
    # Simple reply structure
    reply_to = {
        0: 0,  # alice starts thread
        1: 0,  # bob replies to alice's hello
        2: 1,  # alice replies to bob
        3: 0,  # charlie also replies to alice's hello
        4: 3,  # alice replies to charlie
        5: 2,  # bob replies to alice's "great" message
    }
    
    return Dialogue(
        id="test_dialogue",
        utterances=utterances,
        reply_to=reply_to
    )


def test_data_structures():
    """Test basic data structures."""
    print("Testing data structures...")
    
    dialogue = create_test_dialogue()
    
    assert len(dialogue.utterances) == 6
    assert dialogue.utterances[0].speaker == "alice"
    assert dialogue.utterances[1].text == "Hi alice, how are you?"
    assert dialogue.reply_to[1] == 0
    
    print("✓ Data structures work correctly")


def test_discourse_structures():
    """Test discourse structure construction."""
    print("Testing discourse structures...")
    
    dialogue = create_test_dialogue()
    config = {
        'speaker_utterance': {'enabled': True},
        'speaker_mentioning': {'enabled': True},
        'utterance_distance': {'enabled': True},
        'partial_replying': {'enabled': True}
    }
    
    builder = DiscourseStructureBuilder(config)
    structures = builder.build_all_structures(
        dialogue, 
        context_window=(0, 5),
        partial_replying_edges={(1, 0): 1.0, (2, 1): 1.0}
    )
    
    assert 'speaker_utterance' in structures
    assert 'utterance_distance' in structures
    assert structures['speaker_utterance'].edge_index.size(0) == 2  # [2, num_edges]
    
    print("✓ Discourse structures work correctly")


def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")
    
    # Load minimal config
    config = {
        'model': {
            'bert': {
                'model_name': 'bert-base-uncased',
                'hidden_size': 768,
                'dropout': 0.2
            },
            'egcn': {
                'hidden_sizes': [768, 300],
                'edge_label_embedding_dim': 10,
                'activation': 'relu',
                'dropout': 0.2
            },
            'lstm': {
                'hidden_size': 768,
                'num_layers': 2,
                'bidirectional': True,
                'dropout_feedforward': 0.2,
                'dropout_recurrent': 0.2
            },
            'ffnn': {
                'layer1_hidden_size': 768,
                'layer2_hidden_size': 300,
                'activation': 'relu',
                'dropout': 0.2
            }
        },
        'discourse_structures': {
            'speaker_utterance': {'enabled': True},
            'speaker_mentioning': {'enabled': True},
            'utterance_distance': {'enabled': True},
            'partial_replying': {'enabled': True}
        },
        'data': {
            'context_window_size': 50
        },
        'training': {
            'alpha1': 1.0,
            'alpha2': 0.1,
            'alpha3': 0.05
        }
    }
    
    device = torch.device('cpu')  # Use CPU for testing
    
    try:
        model = DiHRLModel(config).to(device)
        print(f"✓ Model initialized successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test loss function
        loss_fn = HierarchicalRankingLoss()
        print("✓ Loss function initialized correctly")
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise


def test_tokenizer():
    """Test BERT tokenizer."""
    print("Testing BERT tokenizer...")
    
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Test encoding
        text = "Hello world [SEP] How are you?"
        encoded = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
        
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert encoded['input_ids'].size(1) <= 128
        
        print("✓ BERT tokenizer works correctly")
        
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        raise


def test_metrics():
    """Test evaluation metrics."""
    print("Testing evaluation metrics...")
    
    # Create simple test data
    predicted_clusters = [[0, 1, 2], [3, 4, 5]]
    gold_clusters = [[0, 1], [2, 3], [4, 5]]
    
    predicted_links = {0: 0, 1: 0, 2: 1, 3: 3, 4: 3, 5: 4}
    gold_links = {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4}
    
    metrics_calc = DialogueDisentanglementMetrics()
    
    try:
        metrics = metrics_calc.evaluate_all(
            predicted_clusters, gold_clusters,
            predicted_links, gold_links
        )
        
        assert 'adjusted_rand_index' in metrics
        assert 'link_f1' in metrics
        assert 0 <= metrics['adjusted_rand_index'] <= 1
        assert 0 <= metrics['link_f1'] <= 1
        
        print("✓ Evaluation metrics work correctly")
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        raise


def test_gpu_setup():
    """Test GPU setup."""
    print("Testing GPU setup...")
    
    config = {
        'hardware': {'device': 'cuda'},
        'experiment': {'seed': 42, 'deterministic': True}
    }
    
    try:
        device = setup_gpu_training(config)
        print(f"✓ GPU setup completed, using device: {device}")
        
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    except Exception as e:
        print(f"✗ GPU setup failed: {e}")
        # This is not critical for CPU-only systems
        print("  Continuing with CPU...")


def test_model_forward_pass():
    """Test a simple forward pass."""
    print("Testing model forward pass...")
    
    config = {
        'model': {
            'bert': {
                'model_name': 'bert-base-uncased',
                'hidden_size': 768,
                'dropout': 0.2
            },
            'egcn': {
                'hidden_sizes': [768, 300],
                'edge_label_embedding_dim': 10,
                'activation': 'relu',
                'dropout': 0.2
            },
            'lstm': {
                'hidden_size': 768,
                'num_layers': 2,
                'bidirectional': True,
                'dropout_feedforward': 0.2,
                'dropout_recurrent': 0.2
            },
            'ffnn': {
                'layer1_hidden_size': 768,
                'layer2_hidden_size': 300,
                'activation': 'relu',
                'dropout': 0.2
            }
        },
        'discourse_structures': {
            'speaker_utterance': {'enabled': True},
            'speaker_mentioning': {'enabled': True},
            'utterance_distance': {'enabled': True},
            'partial_replying': {'enabled': True}
        },
        'data': {
            'context_window_size': 50,
            'max_utterance_length': 128
        },
        'training': {
            'alpha1': 1.0,
            'alpha2': 0.1,
            'alpha3': 0.05
        }
    }
    
    device = torch.device('cpu')
    model = DiHRLModel(config).to(device)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dialogue = create_test_dialogue()
    
    try:
        with torch.no_grad():
            scores = model(
                dialogue=dialogue,
                current_utterance_id=2,
                candidate_utterance_ids=[0, 1, 2],
                tokenizer=tokenizer,
                partial_replying_edges={(1, 0): 1.0}
            )
        
        assert scores.shape == (3,)  # 3 candidates
        assert torch.all(torch.isfinite(scores))
        
        print("✓ Model forward pass successful")
        print(f"  Output scores: {scores.tolist()}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("DIHRL IMPLEMENTATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_data_structures,
        test_discourse_structures,
        test_tokenizer,
        test_model_initialization,
        test_metrics,
        test_gpu_setup,
        test_model_forward_pass,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 50)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
