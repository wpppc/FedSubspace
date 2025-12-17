
#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup /home/wuqicen/anaconda3/envs/fedsubspace/bin/python main_fedless_alt.py > fedless_alt.log 2>&1 &
echo "Fedless_alt experiment started on GPU 6. Check fedless_alt.log for progress."
