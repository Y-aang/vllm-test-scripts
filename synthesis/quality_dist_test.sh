#!/bin/bash

# ========== 实验配置 ==========
model_name="DeepSeek-R1-Distill-Qwen-1.5B"   # 模型名称（可修改）
dataset_name="Quality"                           # 数据集名称（可修改）
sample_strategy="Distshift"                        # 采样策略（可修改）

script_name="test_script_batch.py"            # 调用的 Python 脚本
param_name="cache_size"                         # 测试参数名
block_size=16                                 # ✅ 每个 block 的 token 数，可灵活修改

# ========== 测试的 Cache Size 列表 ==========
# cache_sizes=(3125 6250 9375 12500 15625 18750)
cache_sizes=(3125)
evictor_types=(LRU_L)

# ========== 循环执行实验 ==========
# 外层循环：cache size
for size in "${cache_sizes[@]}"
do
    # ✅ 使用 block_size 变量
    actual_size=$((size * block_size))
    
    # 内层循环：evictor types
    for evictor_type in "${evictor_types[@]}"
    do
        # 将 evictor_type 转换为小写作为 cache_strategy
        cache_strategy=$(echo "$evictor_type" | tr '[:upper:]' '[:lower:]')
        
        # 根据 cache_strategy 构建输出目录
        output_dir="./test/${model_name}/${dataset_name}/${sample_strategy}/${cache_strategy}"
        
        # 创建输出文件夹
        mkdir -p "$output_dir"
        
        # 设置环境变量
        export VLLM_CUSTOMIZED_EVICTOR_TYPE="$evictor_type"
        
        output_file="${output_dir}/${actual_size}.txt"
        
        echo "🚀 Running with cache_size=${actual_size} tokens (block_size=${block_size}), evictor_type=${evictor_type}, cache_strategy=${cache_strategy}..."
        
        # 调用 Python 脚本并重定向输出
        python "${script_name}" --${param_name} ${actual_size} > "${output_file}" 2>&1
        
        echo "✅ Completed experiment with cache_size=${actual_size}, evictor_type=${evictor_type}, results saved to ${output_file}"
    done
done

echo "🎯 All cache size and evictor type experiments completed."
