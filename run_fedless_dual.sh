#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python -u main_fedless_dual.py > fedless_dual.log 2>&1 &
echo "Fedless Dual (Ablation 2) experiment started on GPU 6. Check fedless_dual.log for progress."
