#!/bin/bash

# 使用ModelScope下载Mistral模型脚本
# 用法: ./download_model.sh

MODEL_NAME="shakechen/Llama-2-7b-hf"
LOCAL_DIR="/home/wuqicen/base_model/Llama2-7B"

echo "🚀 开始使用ModelScope下载llama模型..."
echo "模型: $MODEL_NAME"
echo "保存到: $LOCAL_DIR"

# 创建目录
mkdir -p $LOCAL_DIR

# 使用ModelScope-cli下载
python -c "
from modelscope import snapshot_download
import os
model_dir = snapshot_download('$MODEL_NAME', cache_dir='$LOCAL_DIR')
print(f'✅ 下载完成: {model_dir}')
"

echo "📁 模型保存在: $LOCAL_DIR"