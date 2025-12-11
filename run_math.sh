#!/bin/bash
# Usage: bash run_math.sh [GPU_IDS]
# Example: bash run_math.sh 0
# Example (Multi-card model split): bash run_math.sh 0,1

GPU_IDS=${1:-0}

echo "Running Math Domain Experiment on GPU(s): $GPU_IDS"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the script
/home/wuqicen/anaconda3/envs/fedsubspace/bin/python main_math.py
