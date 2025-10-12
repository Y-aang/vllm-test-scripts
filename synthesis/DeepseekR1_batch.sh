#!/bin/bash

# 设置实验参数
model_name="DeepSeek-R1-Distill-Qwen-1.5B"      # 数据集名称（可修改）
dataset_name="squad"      # 数据集名称（可修改）
sample_strategy="DistShift"  # 采样策略（可修改）
cache_strategy="dbl"       # 缓存策略（可修改："lru" 或 "arc" or "dbl"）

# 输出文件夹路径（自动组织）
output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"

# 如果目录已存在，先删除
# if [ -d "$output_dir" ]; then
#     rm -rf "$output_dir"
#     echo "Deleted existing directory: $output_dir"
# fi

# 创建输出文件夹
mkdir -p "$output_dir"

# 自定义 batch size 序列
batch_sizes=(1 2 4 8 16 32 64 128)

# 遍历 batch size
for batch_size in "${batch_sizes[@]}"
do
    # 生成输出文件路径
    output_file="${output_dir}/${batch_size}.txt"
    
    # 调用 Python 脚本并将输出保存到指定文件
    python test_script_batch.py --batch_size $batch_size > "$output_file" 2>&1
    
    # 打印当前执行情况
    echo "Completed experiment with batch_size=${batch_size}, saved to ${output_file}"
done

echo "All experiments completed."

