#!/bin/bash

# 设置脚本目录
SCRIPT_DIR="/home/shenyang/tests/verify"
RESULT_FILE="/home/shenyang/tests/verify/result/microbench.txt"
PYTHON_SCRIPT="${SCRIPT_DIR}/strategy_speed.py"

# 确保结果目录存在
mkdir -p "$(dirname "$RESULT_FILE")"

# 写入表头
echo "EvictorType,PromptLength,Run,Miss,Hit" > "$RESULT_FILE"

# 定义数组
evictor_types=(DBL)
prompt_lengths=(32 64 128 256 512 1024 2048 4096)
runs=(1 2 3 4 5 6 7 8)

# 最外层循环：Evictor Type (LRU, ARC)
for evictor_type in "${evictor_types[@]}"; do
    echo "=========================================="
    echo "切换到 Evictor Type: $evictor_type"
    echo "=========================================="
    
    export VLLM_CUSTOMIZED_EVICTOR_TYPE="$evictor_type"
    
    # 第二层循环：Prompt Length (256, 512, 1024)
    for prompt_length in "${prompt_lengths[@]}"; do
        echo "----------------------------------------"
        echo "Prompt Length: $prompt_length"
        echo "----------------------------------------"
        
        # 第三层循环：重复实验 8 次
        for run in "${runs[@]}"; do
            echo "运行实验 $run/8 (Evictor: $evictor_type, Prompt Length: $prompt_length)"
            
            # 调用 Python 脚本
            python "$PYTHON_SCRIPT" --prompt_length "$prompt_length" --block_size 16
            
            # 从结果文件中读取最后一行（刚写入的结果）
            if [ -f "$RESULT_FILE" ]; then
                # 获取文件总行数
                total_lines=$(wc -l < "$RESULT_FILE")
                if [ "$total_lines" -gt 1 ]; then
                    # 读取最后一行（去掉换行符）
                    last_line=$(tail -n 1 "$RESULT_FILE" | tr -d '\n')
                    # 如果最后一行是数值格式（包含逗号且不是表头），则替换为带元数据的行
                    if [[ "$last_line" =~ ^[0-9]+\.[0-9]+,[0-9]+\.[0-9]+$ ]]; then
                        # 使用 sed 替换最后一行
                        sed -i "\$s/.*/$evictor_type,$prompt_length,$run,$last_line/" "$RESULT_FILE"
                    fi
                fi
            fi
        done
        
        echo "完成 Prompt Length $prompt_length 的所有实验"
        echo ""
    done
    
    echo "完成 Evictor Type $evictor_type 的所有实验"
    echo ""
done

echo "=========================================="
echo "所有实验完成！"
echo "结果已保存到: $RESULT_FILE"
echo "=========================================="

