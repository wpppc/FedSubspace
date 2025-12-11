#!/bin/bash
# Use two GPUs (e.g., 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python -u main_feddpa.py > feddpa.log 2>&1 &
echo "FedDPA experiment started on GPUs 0,1. Check feddpa.log for progress."
