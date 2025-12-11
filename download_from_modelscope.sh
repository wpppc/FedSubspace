#!/bin/bash

# 定义数据存放的根目录
DATA_ROOT="data/raw_datasets"
mkdir -p $DATA_ROOT

echo "========================================================"
echo "Downloading specific datasets using ModelScope"
echo "========================================================"

# 下载 Commonsense170k
echo "--------------------------------------------------------"
echo "Downloading Commonsense170k..."
modelscope download --dataset deepmath/commonsense_170k --cache_dir $DATA_ROOT/commonsense170k
if [ $? -eq 0 ]; then
    echo "✅ Successfully downloaded Commonsense170k"
else
    echo "❌ Failed to download Commonsense170k"
fi

# 下载 MetaMathQA
echo "--------------------------------------------------------"
echo "Downloading MetaMathQA..."
modelscope download --dataset swift/MetaMathQA --cache_dir $DATA_ROOT/metamathqa
if [ $? -eq 0 ]; then
    echo "✅ Successfully downloaded MetaMathQA"
else
    echo "❌ Failed to download MetaMathQA"
fi

# 下载 competition_math
echo "--------------------------------------------------------"
echo "Downloading competition_math..."
modelscope download --dataset modelscope/competition_math --cache_dir $DATA_ROOT/math
if [ $? -eq 0 ]; then
    echo "✅ Successfully downloaded math"
else
    echo "❌ Failed to download math"
fi

# 检查最终结果
echo "Final status:"
echo "============="
for dataset in commonsense170k metamathqa math; do
    dir_path="$DATA_ROOT/$dataset"
    if [ -d "$dir_path" ] && [ "$(ls -A $dir_path 2>/dev/null)" ]; then
        echo "✅ $dataset: Downloaded"
    else
        echo "❌ $dataset: Missing"
    fi
done