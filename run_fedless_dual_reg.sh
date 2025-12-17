#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python -u main_fedless_dual_reg.py > fedless_dual_reg.log 2>&1 &
echo "Fed-LoRA-Mix (Dual+Reg) experiment started on GPU 1. Check fedless_dual_reg.log for progress."
