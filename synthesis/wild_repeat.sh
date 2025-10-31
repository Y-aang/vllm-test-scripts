#!/bin/bash

# ========== 实验配置 ==========
model_name="DeepSeek-R1-Distill-Qwen-1.5B"   # 模型名称（可修改）
dataset_name="wild"                           # 数据集名称（可修改）
sample_strategy="all"                         # 采样策略（可修改）
cache_strategy="arc"                          # 缓存策略（"lru" / "arc" / "dbl"）

script_name="test_script_batch.py"            # 调用的 Python 脚本
param_name="cache_size"                       # 测试参数名
block_size=16                                 # ✅ 每个 block 的 token 数，可灵活修改

# ========== 输出目录结构 ==========
output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"

# 创建输出文件夹
mkdir -p "$output_dir"

# ========== 固定的 Cache Size（单位：block）==========
size=20000                                    # 你要的固定 cache size（按 block 计），可改

# 计算实际 token 数
actual_size=$((size * block_size))
output_file_base="${output_dir}/${actual_size}"

# ========== 重复执行 8 次 ==========
for i in 1 2 3 4 5 6 7 8
do
    output_file="${output_file_base}_${i}.txt"
    echo "🚀 Running (${i}/4) with cache_size=${actual_size} tokens (block_size=${block_size})..."
    python "${script_name}" --${param_name} ${actual_size} > "${output_file}" 2>&1
    echo "✅ Completed run ${i}/8 with cache_size=${actual_size}, results saved to ${output_file}"
done

echo "🎯 All repeated runs completed. Results are in: ${output_dir}"