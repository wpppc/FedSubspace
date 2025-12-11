#!/bin/bash
# Usage: bash run_commonsense.sh [GPU_IDS]
# Example: bash run_commonsense.sh 2

GPU_IDS=${1:-0}

echo "Running Commonsense Domain Experiment on GPU(s): $GPU_IDS"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Run the script
/home/wuqicen/anaconda3/envs/fedsubspace/bin/python main_commonsense.py
