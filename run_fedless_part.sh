#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python -u main_fedless_part.py > fedless_part.log 2>&1 &
echo "FedLESS-Part experiment started on GPU 5. Check fedless_part.log for progress."
