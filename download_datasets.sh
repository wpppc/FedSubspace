#!/bin/bash

# 检查 huggingface-cli 是否存在
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Please install it via: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

# 设置 HF 镜像 (ModelScope 也可以，但 HF Mirror 对 dataset 支持更直接)
export HF_ENDPOINT=https://hf-mirror.com

# 定义数据存放的根目录
DATA_ROOT="data/raw_datasets"
mkdir -p $DATA_ROOT

echo "========================================================"
echo "Downloading datasets to $DATA_ROOT using $HF_ENDPOINT"
echo "========================================================"

# 定义下载函数
# 使用 --local-dir-use-symlinks False 确保下载的是实际文件而不是缓存链接
download_ds() {
    REPO_ID=$1
    LOCAL_DIR_NAME=$2
    TARGET_DIR="$DATA_ROOT/$LOCAL_DIR_NAME"
    
    echo "--------------------------------------------------------"
    echo "Downloading $REPO_ID to $TARGET_DIR ..."
    
    huggingface-cli download --repo-type dataset \
        "$REPO_ID" \
        --local-dir "$TARGET_DIR" \
        --local-dir-use-symlinks False \
        --resume-download \
        --quiet
        
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $LOCAL_DIR_NAME"
    else
        echo "Failed to download $LOCAL_DIR_NAME"
    fi
}

# === 1. Math Reasoning ===
# Training
download_ds "MetaMathQA/MetaMathQA" "metamathqa"
# Evaluation
download_ds "gsm8k" "gsm8k"
download_ds "hendrycks/competition_math" "math"

# === 2. Code Generation ===
# Training (CodeFeedback105k usually refers to the filtered instruction set)
download_ds "m-a-p/CodeFeedback-Filtered-Instruction" "codefeedback"
# Evaluation
download_ds "openai_humaneval" "humaneval"
download_ds "mbpp" "mbpp"

# === 3. Commonsense Reasoning ===
# Training
download_ds "TIGER-Lab/Commonsense170k" "commonsense170k"
# Evaluation
download_ds "google/boolq" "boolq"
download_ds "piqa" "piqa"
download_ds "social_i_qa" "siqa"
download_ds "Rowan/hellaswag" "hellaswag"
download_ds "winogrande" "winogrande"
download_ds "ai2_arc" "ai2_arc"
download_ds "openbookqa" "openbookqa"

echo "========================================================"
echo "All downloads completed."
echo "Datasets are located in: $(pwd)/$DATA_ROOT"
echo "Please ensure your config file points to these local paths."
echo "========================================================"
