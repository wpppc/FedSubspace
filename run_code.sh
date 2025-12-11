#!/bin/bash
# Usage: bash run_code.sh [GPU_IDS]
# Example: bash run_code.sh 1

GPU_IDS=${1:-0}

echo "Running Code Domain Experiment on GPU(s): $GPU_IDS"
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Run the script
/home/wuqicen/anaconda3/envs/fedsubspace/bin/python main_code.py
