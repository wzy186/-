#!/bin/bash

# Transformer from Scratch - Reproducible Experiment Runner
# This script reproduces the exact experiments with fixed random seeds

set -e  # Exit on error

echo "==========================================="
echo "Transformer from Scratch - Reproducible Experiments"
echo "==========================================="

# Fixed random seeds for reproducibility
MAIN_SEED=42
ABLATION_SEED=42

echo "Using fixed random seeds:"
echo "  Main experiment: $MAIN_SEED"
echo "  Ablation study: $ABLATION_SEED"

# Create results directory
mkdir -p ../results

# Setup environment
echo "Setting up Python environment..."
conda create -n transformer python=3.10 -y
conda activate transformer

echo "Installing dependencies..."
pip install -r ../requirements.txt

echo "Environment setup completed!"

# Run main training with exact seed
echo "Starting main training with seed $MAIN_SEED..."
cd ../src
python train.py --config ../configs/base.yaml --seed $MAIN_SEED --device cuda

echo "Main training completed!"

# Run ablation study with exact seed
echo "Starting ablation study with seed $ABLATION_SEED..."
python ablation_study.py --config ../configs/base.yaml --seed $ABLATION_SEED --device cuda

echo "Ablation study completed!"

# Generate final report
echo "Generating results report..."
python generate_report.py

echo "==========================================="
echo "All experiments completed successfully!"
echo "Results saved to: ../results/"
echo "To verify reproducibility, compare with expected values in README.md"
echo "==========================================="