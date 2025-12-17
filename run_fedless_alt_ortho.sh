#!/bin/bash
# Run FedALT with Orthogonality Regularization
# Usage: ./run_alt_ortho.sh [gpu_id]

GPU_ID=${1:-0}

echo "Running FedALT+Ortho on GPU $GPU_ID..."

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Ensure output directory exists
mkdir -p outputs/fedless+alt+ortho

# Run the script
/home/wuqicen/anaconda3/envs/fedsubspace/bin/python main_fedless_alt_ortho.py > fedless_alt_ortho.log 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Logs are being written to fedless_alt_ortho.log"
echo "To follow logs: tail -f fedless_alt_ortho.log"
