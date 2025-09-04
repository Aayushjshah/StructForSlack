"""
Main training script for DiHRL model.
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer import DiHRLTrainer, setup_gpu_training
from src.utils.data_utils import IRCDataLoader
from src.evaluation.metrics import DialogueDisentanglementMetrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data_paths(config: dict) -> dict:
    """Set up data paths relative to the current directory."""
    # Update data path to be relative to current directory
    data_path = config['data']['data_path']
    if not os.path.isabs(data_path):
        config['data']['data_path'] = os.path.abspath(data_path)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train DiHRL model for dialogue disentanglement')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use (overrides config)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced data'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = setup_data_paths(config)
    
    # Override GPU setting if specified
    if args.gpu is not None:
        config['hardware']['device'] = 'cuda' if args.gpu >= 0 else 'cpu'
        if args.gpu >= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Set up GPU training environment
    device = setup_gpu_training(config)
    
    # Create necessary directories
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(
        config['logging']['log_dir'],
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting DiHRL training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Data path: {config['data']['data_path']}")
    
    try:
        # Load data
        logger.info("Loading IRC data...")
        data_loader = IRCDataLoader(config['data']['data_path'])
        
        # Load training and validation data
        train_dialogues = data_loader.load_dialogues('train')
        dev_dialogues = data_loader.load_dialogues('dev')
        
        logger.info(f"Loaded {len(train_dialogues)} training dialogues")
        logger.info(f"Loaded {len(dev_dialogues)} validation dialogues")
        
        # Debug mode: use subset of data
        if args.debug:
            train_dialogues = train_dialogues[:10]
            dev_dialogues = dev_dialogues[:5]
            logger.info(f"Debug mode: Using {len(train_dialogues)} train, {len(dev_dialogues)} dev dialogues")
        
        # Initialize trainer
        trainer = DiHRLTrainer(config, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        history = trainer.train(train_dialogues, dev_dialogues)
        
        logger.info("Training completed successfully!")
        
        # Save final results
        results_file = os.path.join(
            config['logging']['log_dir'],
            f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        
        with open(results_file, 'w') as f:
            yaml.dump({
                'config': config,
                'training_history': history,
                'best_metric': trainer.best_metric
            }, f)
        
        logger.info(f"Results saved to {results_file}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
