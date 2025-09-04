"""
Main evaluation script for DiHRL model.
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import json
from datetime import datetime
from transformers import BertTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.dihrl_model import DiHRLModel
from src.models.easy_first_decoder import EasyFirstDecoder, DecodingEvaluator
from src.utils.data_utils import IRCDataLoader
from src.evaluation.metrics import DialogueDisentanglementMetrics, links_to_clusters
from src.training.trainer import setup_gpu_training


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, config: dict, device: torch.device) -> DiHRLModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state
    model = DiHRLModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_dialogues(model: DiHRLModel, dialogues, tokenizer: BertTokenizer, 
                      config: dict, use_easy_first: bool = True) -> dict:
    """Evaluate model on a set of dialogues."""
    metrics_calculator = DialogueDisentanglementMetrics()
    decoder = EasyFirstDecoder(model, tokenizer, config)
    
    all_predicted_links = {}
    all_gold_links = {}
    
    results = {
        'dialogue_results': [],
        'aggregate_metrics': {},
        'decoding_strategy': 'easy_first' if use_easy_first else 'sequential'
    }
    
    for i, dialogue in enumerate(dialogues):
        try:
            # Decode dialogue
            if use_easy_first:
                predicted_links = decoder.decode_dialogue(dialogue)
            else:
                predicted_links = decoder.decode_dialogue_sequential(dialogue)
            
            # Store results
            dialogue_id = dialogue.id
            all_predicted_links[dialogue_id] = predicted_links
            all_gold_links[dialogue_id] = dialogue.reply_to
            
            # Convert to clusters for evaluation
            pred_clusters = links_to_clusters(predicted_links)
            gold_clusters = dialogue.sessions if dialogue.sessions else links_to_clusters(dialogue.reply_to)
            
            # Compute metrics for this dialogue
            dialogue_metrics = metrics_calculator.evaluate_all(
                predicted_clusters=pred_clusters,
                gold_clusters=gold_clusters,
                predicted_links=predicted_links,
                gold_links=dialogue.reply_to
            )
            
            results['dialogue_results'].append({
                'dialogue_id': dialogue_id,
                'num_utterances': len(dialogue.utterances),
                'num_sessions': len(gold_clusters),
                'metrics': dialogue_metrics
            })
            
            print(f"Evaluated dialogue {i+1}/{len(dialogues)}: {dialogue_id}")
            
        except Exception as e:
            print(f"Error evaluating dialogue {dialogue.id}: {e}")
            continue
    
    # Compute aggregate metrics
    if all_predicted_links and all_gold_links:
        # Flatten all predictions
        flat_pred_links = {}
        flat_gold_links = {}
        
        for dialogue_id in all_predicted_links:
            flat_pred_links.update(all_predicted_links[dialogue_id])
            flat_gold_links.update(all_gold_links[dialogue_id])
        
        # Convert to clusters
        pred_clusters = links_to_clusters(flat_pred_links)
        gold_clusters = links_to_clusters(flat_gold_links)
        
        # Compute aggregate metrics
        aggregate_metrics = metrics_calculator.evaluate_all(
            predicted_clusters=pred_clusters,
            gold_clusters=gold_clusters,
            predicted_links=flat_pred_links,
            gold_links=flat_gold_links
        )
        
        results['aggregate_metrics'] = aggregate_metrics
    
    return results


def run_ablation_study(model: DiHRLModel, test_dialogues, tokenizer: BertTokenizer, 
                      config: dict) -> dict:
    """Run ablation study comparing different components."""
    print("Running ablation study...")
    
    # Test easy-first vs sequential decoding
    easy_first_results = evaluate_dialogues(
        model, test_dialogues[:10], tokenizer, config, use_easy_first=True
    )
    
    sequential_results = evaluate_dialogues(
        model, test_dialogues[:10], tokenizer, config, use_easy_first=False
    )
    
    # Compare confidence trajectories
    decoder_evaluator = DecodingEvaluator(model, tokenizer, config)
    confidence_analysis = {}
    
    for i, dialogue in enumerate(test_dialogues[:5]):
        try:
            trajectories = decoder_evaluator.analyze_decoding_confidence(dialogue)
            confidence_analysis[dialogue.id] = trajectories
        except Exception as e:
            print(f"Error analyzing dialogue {dialogue.id}: {e}")
    
    return {
        'easy_first_decoding': easy_first_results['aggregate_metrics'],
        'sequential_decoding': sequential_results['aggregate_metrics'],
        'confidence_trajectories': confidence_analysis
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate DiHRL model for dialogue disentanglement')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['dev', 'test'],
        help='Data split to evaluate on'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (default: auto-generated)'
    )
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run ablation study'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Evaluate on subset of dialogues (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override GPU setting
    if args.gpu is not None:
        config['hardware']['device'] = 'cuda' if args.gpu >= 0 else 'cpu'
        if args.gpu >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Set up device
    device = setup_gpu_training(config)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting DiHRL evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Device: {device}")
    
    try:
        # Load data
        logger.info("Loading IRC data...")
        data_loader = IRCDataLoader(config['data']['data_path'])
        test_dialogues = data_loader.load_dialogues(args.split)
        
        logger.info(f"Loaded {len(test_dialogues)} dialogues for evaluation")
        
        # Use subset if specified
        if args.subset:
            test_dialogues = test_dialogues[:args.subset]
            logger.info(f"Using subset of {len(test_dialogues)} dialogues")
        
        # Load model
        logger.info("Loading trained model...")
        model = load_model(args.model, config, device)
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(config['model']['bert']['model_name'])
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluate_dialogues(model, test_dialogues, tokenizer, config)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        aggregate_metrics = results['aggregate_metrics']
        if aggregate_metrics:
            print(f"Variation Information: {aggregate_metrics.get('variation_information', 0.0):.4f}")
            print(f"Adjusted Rand Index: {aggregate_metrics.get('adjusted_rand_index', 0.0):.4f}")
            print(f"One-to-One: {aggregate_metrics.get('one_to_one', 0.0):.4f}")
            print(f"Normalized Mutual Information: {aggregate_metrics.get('normalized_mutual_information', 0.0):.4f}")
            print(f"Local-3: {aggregate_metrics.get('local_k', 0.0):.4f}")
            print(f"Shen-F1: {aggregate_metrics.get('shen_f1', 0.0):.4f}")
            print(f"Cluster F1: {aggregate_metrics.get('cluster_f1', 0.0):.4f}")
            print(f"Link F1: {aggregate_metrics.get('link_f1', 0.0):.4f}")
        
        # Run ablation study if requested
        if args.ablation:
            logger.info("Running ablation study...")
            ablation_results = run_ablation_study(model, test_dialogues, tokenizer, config)
            results['ablation_study'] = ablation_results
            
            print("\n" + "="*50)
            print("ABLATION STUDY RESULTS")
            print("="*50)
            
            easy_first_f1 = ablation_results['easy_first_decoding'].get('cluster_f1', 0.0)
            sequential_f1 = ablation_results['sequential_decoding'].get('cluster_f1', 0.0)
            
            print(f"Easy-First Decoding F1: {easy_first_f1:.4f}")
            print(f"Sequential Decoding F1: {sequential_f1:.4f}")
            print(f"Improvement: {easy_first_f1 - sequential_f1:.4f}")
        
        # Save results
        if args.output:
            output_file = args.output
        else:
            output_file = f"evaluation_results_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
