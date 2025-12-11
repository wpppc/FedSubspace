
#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python main_alt.py > fed_alt.log 2>&1 &
echo "FedALT experiment started on GPU 6. Check fed_alt.log for progress."
