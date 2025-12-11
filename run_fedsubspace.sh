#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python -u main_flan.py > fedsubspace.log 2>&1 &
echo "Flan experiment started on GPU 1. Check fedsubspace.log for progress."
