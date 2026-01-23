# Architectural Trade-offs in Small Language Models Under Compute Constraints

This repository contains code and experiments for a systematic empirical study of small language models under strict compute constraints, analyzing how architectural choices and training budget interact to determine performance.

**Paper**: [Architectural Trade-offs in Small Language Models Under Compute Constraints](https://arxiv.org/abs/2512.20877)

**Author**: Shivraj Singh Bhatti (ssbhatti@umass.edu)

## Overview

This project compares different language model architectures (Linear, MLP, Self-Attention, and Transformer) on character-level and word-level language modeling tasks. Starting from a simple linear next-token predictor, we progressively introduce nonlinearities, self-attention, and multi-layer transformer architectures, evaluating each on:

- **Character-level modeling**: Tiny Shakespeare dataset
- **Word-level modeling**: Penn Treebank (PTB) and WikiText-2 datasets

Models are compared using test negative log-likelihood (NLL), parameter count, and approximate training FLOPs to characterize accuracy-efficiency trade-offs.

## Key Findings

- **Attention-based models dominate MLPs** in per-FLOP efficiency even at small scale
- **Increasing depth or context without sufficient optimization** can degrade performance
- **Modern LLM techniques may not transfer** to small-model regimes (e.g., RoPE positional embeddings underperformed learned embeddings)
- **Nonlinear models dramatically outperform** the linear baseline
- **Capacity must match compute**: additional parameters need sufficient optimization steps

## Repository Structure

```
language_model_architecture_comparison/
├── language_model_architecture_comparison.ipynb  # Main Jupyter notebook with all experiments
├── datasets/                                      # Dataset files
│   ├── tiny_shakespeare/                         # Character-level dataset
│   ├── ptb/                                      # Penn Treebank (word-level)
│   └── wikitext-2/                               # WikiText-2 (word-level)
├── images/                                       # Generated plots and visualizations
│   ├── tiny_train_val_loss.png                  # Training curves for Tiny Shakespeare
│   ├── tiny_loss_vs_flops.png                   # Efficiency comparison
│   ├── ptb_loss.png                              # PTB training dynamics
│   ├── wikitext2_loss.png                       # WikiText-2 training dynamics
│   └── ...                                       # Additional hyperparameter sweep plots
└── README.md                                     # This file
```

## Setup

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- CUDA-capable GPU (optional, but recommended for faster training)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/shivraj-S-bhatti/language_model_architecture_comparison.git
cd language_model_architecture_comparison
```

2. Install required dependencies:
```bash
pip install torch numpy matplotlib jupyter
```

Or using conda:
```bash
conda install pytorch numpy matplotlib jupyter -c pytorch
```

## Usage

### Running Experiments

1. Open the Jupyter notebook:
```bash
jupyter notebook language_model_architecture_comparison.ipynb
```

2. The notebook is organized into sections:
   - **Data Loading**: Loads and tokenizes Tiny Shakespeare dataset
   - **Model Definitions**: Implements Linear, MLP, Self-Attention, and Transformer architectures
   - **Training**: Trains each model with appropriate hyperparameters
   - **Hyperparameter Sweeps**: Explores context length, hidden dimensions, attention heads, and layer depth
   - **Evaluation**: Computes test NLL, parameter counts, and approximate FLOPs
   - **Visualization**: Generates plots comparing models

3. Update the dataset path if needed:
   - The notebook uses `BASE_DIR = "/kaggle/input/dataset-hw6/datasets"` by default
   - For local use, update this to point to the `datasets/` folder in this repository

### Model Architectures

The repository implements four progressively complex architectures:

<div align="center">
  <img src="images/model_architectures.png" alt="Model Architecture Comparison" width="90%"/>
  <p><em>LeNet-style architecture diagrams showing the four model designs</em></p>
</div>

1. **Linear Model**: Multinomial logistic regression over concatenated embeddings
2. **MLP**: Two hidden layers with ReLU activations and dropout
3. **Self-Attention**: Single transformer block with causal multi-head attention
4. **Transformer**: Multi-layer transformer with 3 layers (configurable)

### Hyperparameter Sweeps

The notebook includes hyperparameter sweeps for:
- **Linear**: Context length `{32, 64, 128}`
- **MLP**: Hidden dimension `{128, 256, 512}`
- **Self-Attention**: Number of heads `{1, 2, 4}`
- **Transformer**: Number of layers `{2, 3, 4}`

## Datasets

### Tiny Shakespeare
- **Type**: Character-level language modeling
- **Size**: ~1 MB of text
- **Splits**: Train/validation/test provided
- **Tokenization**: Character-level vocabulary

### Penn Treebank (PTB)
- **Type**: Word-level language modeling
- **Vocabulary**: 9,999 tokens (training split only)
- **Context**: 64 words

### WikiText-2
- **Type**: Word-level language modeling
- **Vocabulary**: 33,277 tokens (training split only)
- **Context**: 64 words

## Results

### Training Curves

#### Tiny Shakespeare - All Architectures
![Training and Validation Loss](images/tiny_train_val_loss.png)

#### Word-Level Models
<div align="center">
  <img src="images/ptb_loss.png" alt="PTB Training Dynamics" width="60%"/>
  <p><em>PTB Training and Validation Loss</em></p>
</div>

<div align="center">
  <img src="images/wikitext2_loss.png" alt="WikiText-2 Training Dynamics" width="60%"/>
  <p><em>WikiText-2 Training and Validation Loss</em></p>
</div>

### Efficiency Comparisons

#### Character-Level (Tiny Shakespeare)
<div align="center">
  <img src="images/tiny_loss_vs_flops.png" alt="Test NLL vs Training FLOPs" width="60%"/>
</div>

#### Word-Level Models
<div align="center">
  <img src="images/wordlevel_loss_vs_flops.png" alt="Word-Level Test NLL vs Training FLOPs" width="60%"/>
</div>

### Hyperparameter Sweeps

#### Linear Model: Context Length
<div align="center">
  <img src="images/linear_ctx_vs_loss.png" alt="Linear Model Context Length Sweep" width="60%"/>
</div>

#### MLP: Hidden Dimension
<div align="center">
  <img src="images/mlp_hidden_vs_loss.png" alt="MLP Hidden Dimension Sweep" width="60%"/>
</div>

#### Self-Attention: Number of Heads
<div align="center">
  <img src="images/attn_heads_vs_loss.png" alt="Self-Attention Heads Sweep" width="60%"/>
</div>

#### Transformer: Number of Layers
<div align="center">
  <img src="images/trans_layers_vs_loss.png" alt="Transformer Layers Sweep" width="60%"/>
</div>

### RoPE Positional Encoding Comparison

#### Training Dynamics
<div align="center">
  <img src="images/tiny_learned_vs_rope_loss.png" alt="Learned vs RoPE Loss" width="60%"/>
</div>

#### Efficiency Comparison
<div align="center">
  <img src="images/tiny_learned_vs_rope_flops.png" alt="Learned vs RoPE FLOPs" width="60%"/>
</div>

### Key Results Summary

| Model | Parameters | Test NLL (Tiny Shakespeare) |
|-------|-----------|----------------------------|
| Linear | 1,073,345 | 3.05 |
| MLP | 4,285,377 | 2.32 |
| Self-Attention | 231,617 | 2.13 |
| Transformer | 430,785 | 2.01 |

Attention-based models achieve better performance with fewer parameters and lower compute requirements.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{bhatti2024architectural,
  title={Architectural Trade-offs in Small Language Models Under Compute Constraints},
  author={Bhatti, Shivraj Singh},
  year={2025},
  url={https://github.com/shivraj-S-bhatti/language_model_architecture_comparison}
}
```

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or issues, please contact:
- **Email**: ssbhatti@umass.edu
- **GitHub**: [@shivraj-S-bhatti](https://github.com/shivraj-S-bhatti)

