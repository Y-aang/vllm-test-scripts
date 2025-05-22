#!/bin/bash

Prompt_lengths=(704 736 768 800 832 864 896 928 960 992 1024 1056 1088 1120 1152 1184 1216 1248 1280 1312 1344 1376 1408 1440 1472 1504 1536 1568 1600 1632 1664)
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
