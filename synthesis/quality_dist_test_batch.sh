#!/bin/bash

# ========== 实验配置 ==========
model_name="Qwen14B_WikiQA"   # 模型名称（可修改）
dataset_name="Quality"                           # 数据集名称（可修改）
sample_strategy="Distshift"                        # 采样策略（可修改）

script_name="test_script_batch.py"            # 调用的 Python 脚本
param_name="cache_size"                         # 测试参数名
block_size=16                                 # ✅ 每个 block 的 token 数，可灵活修改

# ========== 测试配置 ==========
cache_size=18750                                    # 固定的 cache size（blocks）
# cache_size=12500                                    # 固定的 cache size（blocks）
actual_cache_size=$((cache_size * block_size))     # 实际的 cache size（tokens）
batch_sizes=(8 1)                            # batch size 列表（按顺序）
evictor_types=(LRU ARC DBL)                        # evictor types 列表

# ========== 循环执行实验 ==========
# 最外层循环：batch size
for batch_size in "${batch_sizes[@]}"
do
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
        
        output_file="${output_dir}/${actual_cache_size}_${batch_size}.txt"
        
        echo "🚀 Running with cache_size=${actual_cache_size} tokens (block_size=${block_size}), batch_size=${batch_size}, evictor_type=${evictor_type}, cache_strategy=${cache_strategy}..."
        
        # 调用 Python 脚本并重定向输出
        python "${script_name}" --${param_name} ${actual_cache_size} --batch_size ${batch_size} > "${output_file}" 2>&1
        
        echo "✅ Completed experiment with cache_size=${actual_cache_size}, batch_size=${batch_size}, evictor_type=${evictor_type}, results saved to ${output_file}"
    done
done

echo "🎯 All batch size and evictor type experiments completed."
