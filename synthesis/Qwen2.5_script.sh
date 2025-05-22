#!/bin/bash

# 设置实验参数
model_name="Qwen2.5-1.5B-Instruct"      # 数据集名称（可修改）
dataset_name="squad"      # 数据集名称（可修改）
sample_strategy="hotspot"  # 采样策略（可修改）
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

custom_sequence=(10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100)
# custom_sequence=(60 65 70 75 80 85 90 95 100)
# 循环运行 CP_ratio 从 5 到 50，每次间隔 5
for cp_ratio in "${custom_sequence[@]}"
do
    # 生成输出文件路径
    output_file="${output_dir}/${cp_ratio}.txt"
    
    # 调用 Python 脚本并将输出保存到指定文件
    python test_script.py --cp_ratio $cp_ratio > "$output_file" 2>&1
    
    # 打印当前执行情况
    echo "Completed experiment with CP_ratio=${cp_ratio}, saved to ${output_file}"
done

echo "All experiments completed."
