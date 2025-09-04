# DiHRL: Discourse-Aware Encoding and Hierarchical Ranking Loss for Dialogue Disentanglement

This repository implements the DiHRL model from the paper "Revisiting Conversation Discourse for Dialogue Disentanglement" for dialogue disentanglement on the Ubuntu IRC dataset.

## Overview

DiHRL enhances dialogue disentanglement by taking full advantage of dialogue discourse characteristics through:

1. **Four Types of Discourse Structures**:
   - Speaker-Utterance Structure (GS)
   - Speaker-Mentioning Structure (GM)  
   - Utterance-Distance Structure (GD)
   - Partial-Replying Structure (GR)

2. **Hierarchical Ranking Loss (HRL)**: Groups utterances into different discourse levels for better optimization

3. **Easy-First Decoding**: Non-sequential decoding strategy that processes easy decisions first

## Features

- ✅ **GPU Support**: Full NVIDIA GPU compatibility with mixed precision training
- ✅ **BERT Integration**: Uses BERT-base for utterance encoding
- ✅ **Edge-Aware GCN**: Custom graph neural networks for discourse structure modeling
- ✅ **Hierarchical Loss**: Implementation of L1, L2, L3 losses from the paper
- ✅ **Easy-First Decoding**: Algorithm 1 from the paper
- ✅ **Comprehensive Metrics**: All evaluation metrics from the paper
- ✅ **Ablation Studies**: Built-in ablation study capabilities

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (optional but recommended)
- 16GB+ RAM recommended

### Setup

1. **Clone and navigate to the implementation**:
```bash
cd DiHRL_Implementation
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up the IRC dataset path**:
Make sure the IRC dataset is located at `../jkkummerfeld-irc-disentanglement-82ed04f/data` or update the path in `configs/config.yaml`.

## Dataset Structure

The expected dataset structure:
```
jkkummerfeld-irc-disentanglement-82ed04f/
└── data/
    ├── train/
    ├── dev/ 
    ├── test/
    ├── gold.train.graphs.txt
    ├── gold.dev.graphs.txt
    ├── gold.test.graphs.txt
    ├── gold.train.clusters.txt
    ├── gold.dev.clusters.txt
    └── gold.test.clusters.txt
```

## Usage

### Training

**Basic training**:
```bash
python train.py
```

**GPU training**:
```bash
python train.py --gpu 0
```

**Debug mode** (small dataset):
```bash
python train.py --debug --gpu 0
```

**Resume from checkpoint**:
```bash
python train.py --resume checkpoints/checkpoint_epoch_5.pt
```

### Evaluation

**Evaluate on test set**:
```bash
python evaluate.py --model checkpoints/best_model.pt --split test
```

**Run ablation study**:
```bash
python evaluate.py --model checkpoints/best_model.pt --split test --ablation
```

**Evaluate subset for debugging**:
```bash
python evaluate.py --model checkpoints/best_model.pt --split dev --subset 10
```

## Configuration

The main configuration file is `configs/config.yaml`. Key settings:

### Model Configuration
```yaml
model:
  bert:
    model_name: "bert-base-uncased"
    hidden_size: 768
    dropout: 0.2
  egcn:
    num_layers: 2
    hidden_sizes: [768, 300]
  lstm:
    hidden_size: 768
    bidirectional: true
```

### Training Configuration
```yaml
training:
  learning_rate_bert: 6e-6
  learning_rate_non_bert: 1e-5
  num_epochs: 4
  alpha1: 1.0    # L1 loss weight
  alpha2: 0.1    # L2 loss weight  
  alpha3: 0.05   # L3 loss weight
```

### Hardware Configuration
```yaml
hardware:
  device: "cuda"
  mixed_precision: true
  num_gpus: 1
```

## Project Structure

```
DiHRL_Implementation/
├── configs/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── models/
│   │   ├── dihrl_model.py       # Main DiHRL model
│   │   ├── discourse_structures.py  # Discourse graph construction
│   │   ├── egcn.py              # Edge-aware GCN implementation
│   │   └── easy_first_decoder.py    # Easy-first decoding algorithm
│   ├── training/
│   │   └── trainer.py           # Training loop with GPU support
│   ├── evaluation/
│   │   └── metrics.py           # All evaluation metrics
│   └── utils/
│       └── data_utils.py        # Data loading and processing
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Model Architecture

### DiHRL Model Components

1. **BERT Encoder**: Encodes utterance pairs
2. **Discourse Structure Builder**: Constructs four types of graphs
3. **EGCN Layers**: Process discourse structures
4. **BiLSTM**: Temporal modeling with DropConnect
5. **FFNN**: Final prediction layer

### Discourse Structures

- **Speaker-Utterance (GS)**: Connects utterances from same speaker
- **Speaker-Mentioning (GM)**: Links utterances that mention speakers
- **Utterance-Distance (GD)**: Gaussian-weighted distance relationships
- **Partial-Replying (GR)**: Previously established reply relations

### Hierarchical Ranking Loss

- **L1**: Parent vs all candidates
- **L2**: Ancestors vs {ancestors, inner-session, outer-session}
- **L3**: Inner-session vs {inner-session, outer-session}

## Evaluation Metrics

The implementation includes all metrics from the paper:

### Cluster-Level Metrics
- Variation of Information (VI)
- Adjusted Rand Index (ARI)
- One-to-One (1-1)
- Normalized Mutual Information (NMI)
- Local-k (k=3)
- Shen-F1
- Cluster Exact Match (Precision, Recall, F1)

### Pairwise Metrics
- Link Exact Match (Precision, Recall, F1)

### Additional Metrics
- Partial-ARI for subset analysis

## Expected Results

Based on the paper, DiHRL should achieve state-of-the-art performance on Ubuntu IRC:

| Metric | Expected Score |
|--------|---------------|
| VI | ~94.2 |
| ARI | ~81.1 |
| 1-1 | ~84.2 |
| NMI | ~91.9 |
| Cluster F1 | ~48.9 |
| Link F1 | ~75.1 |

## GPU Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **Multi-GPU**: Supported through PyTorch DataParallel

## Memory Usage

- **Training**: ~12-16GB GPU memory with mixed precision
- **Evaluation**: ~6-8GB GPU memory
- **CPU**: 16GB+ RAM recommended for large dialogues

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in config.yaml
   data:
     batch_size: 1
   ```

2. **Data Loading Errors**:
   ```bash
   # Check dataset path in config.yaml
   data:
     data_path: "/path/to/jkkummerfeld-irc-disentanglement-82ed04f/data"
   ```

3. **Import Errors**:
   ```bash
   # Make sure you're in the DiHRL_Implementation directory
   cd DiHRL_Implementation
   python train.py
   ```

### Debug Mode

Use debug mode for testing:
```bash
python train.py --debug --gpu 0
python evaluate.py --model checkpoints/best_model.pt --subset 5
```

## Implementation Details

### Key Features from Paper

- **Equation 3**: Utterance-distance computation with Gaussian weighting
- **Equation 10**: Discourse structure fusion through element-wise addition  
- **Algorithm 1**: Easy-first decoding implementation
- **Hierarchical Loss**: L1, L2, L3 loss computation as in Equations 15-17

### Extensions

- Mixed precision training for faster GPU computation
- Comprehensive logging with Weights & Biases integration
- Modular design for easy experimentation
- Built-in ablation study tools

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{li2024revisiting,
  title={Revisiting Conversation Discourse for Dialogue Disentanglement},
  author={Li, Bobo and Fei, Hao and Li, Fei and Wu, Shengqiong and Liao, Lizi and Wei, Yinwei and Chua, Tat-Seng and Ji, Donghong},
  journal={ACM Transactions on Information Systems},
  volume={43},
  number={1},
  pages={1--34},
  year={2024},
  publisher={ACM}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper's license and the IRC dataset license for usage terms.

## Contributing

This implementation follows the paper specifications closely. For improvements or bug fixes:

1. Ensure changes align with the paper's methodology
2. Test thoroughly on the IRC dataset
3. Maintain GPU compatibility
4. Update documentation as needed

## Contact

For questions about this implementation, please refer to the original paper or create an issue in the repository.
