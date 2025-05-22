#!/bin/bash

Prompt_lengths=(32 64 128 256 512 1024 2048 4096 8192)
RESULTS_FILE="./result/results.txt"
LATENCIES_FILE="./result/latencies.txt"

# 清空旧的结果文件
echo "Block Size, Mean Latency, Std Dev" > $RESULTS_FILE
echo "Block Size, Latencies" > $LATENCIES_FILE

for prompt_length in "${Prompt_lengths[@]}"
do
    echo "Running experiment with prompt_length=$prompt_length..."

    python positional_dependency.py $prompt_length $RESULTS_FILE $LATENCIES_FILE

    echo "Completed experiment with prompt_length=$prompt_length"
    echo ""
done

echo "All experiments finished! Results saved in $RESULTS_FILE and $LATENCIES_FILE"
