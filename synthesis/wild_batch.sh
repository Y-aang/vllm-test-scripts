#!/bin/bash

# ========== 实验配置 ==========
model_name="DeepSeek-R1-Distill-Qwen-1.5B"   # 模型名称（可修改）
dataset_name="wild"                           # 数据集名称（可修改）
sample_strategy="all"                        # 采样策略（可修改）
cache_strategy="arc"                          # 缓存策略（"lru" / "arc" / "dbl"）

script_name="test_script_batch.py"            # 调用的 Python 脚本
param_name="cache_size"                         # 测试参数名
block_size=16                                 # ✅ 每个 block 的 token 数，可灵活修改

# ========== 输出目录结构 ==========
output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"

# 创建输出文件夹
mkdir -p "$output_dir"

# ========== 测试的 Cache Size 列表 ==========
cache_sizes=(2000 5000 10000 15000 20000 38000)
cache_sizes=(38000 20000 15000 10000 5000 2000)
# cache_sizes=(2000 5000 10000)
# cache_sizes=(20000)

# ========== 循环执行实验 ==========
for size in "${cache_sizes[@]}"
do
    # ✅ 使用 block_size 变量
    actual_size=$((size * block_size))
    output_file="${output_dir}/${actual_size}.txt"

    echo "🚀 Running with cache_size=${actual_size} tokens (block_size=${block_size})..."
    
    # 调用 Python 脚本并重定向输出
    python "${script_name}" --${param_name} ${actual_size} > "${output_file}" 2>&1
    
    echo "✅ Completed experiment with cache_size=${actual_size}, results saved to ${output_file}"
done

echo "🎯 All cache size experiments completed. Results are in: ${output_dir}"
