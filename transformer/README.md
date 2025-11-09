# Transformer from Scratch

A complete implementation of the Transformer architecture from scratch for language modeling tasks, with reproducible experiments and comprehensive ablation studies.

## 📋 Project Overview

This project implements the Transformer architecture as described in "Attention Is All You Need" (Vaswani et al., 2017) from scratch. The implementation includes:

- Multi-head self-attention mechanism
- Positional encoding (sinusoidal)
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Complete training pipeline with ablation studies

## 🛠️ Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores (Intel i5 or equivalent)
- **RAM**: 8GB
- **Storage**: 2GB free space
- **GPU**: Optional, but recommended for faster training

### Recommended Setup
- **CPU**: 8+ cores (Intel i7 or equivalent)
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/3080, V100, A100)
- **Storage**: SSD recommended for faster data loading

### Tested Configurations
| Configuration | Training Time | Memory Usage |
|---------------|---------------|--------------|
| CPU: Intel i7-12700K | ~45 minutes | 6GB RAM |
| GPU: NVIDIA RTX 3080 (10GB) | ~10 minutes | 4.2GB VRAM |
| GPU: NVIDIA V100 (16GB) | ~8 minutes | 3.8GB VRAM |
| GPU: NVIDIA A100 (40GB) | ~7 minutes | 3.5GB VRAM |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**:
```bash
git clone (https://github.com/wzy186/-/tree/main/transformer)

cd transformer-from-scratch
