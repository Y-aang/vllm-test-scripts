#!/bin/bash

BLOCK_SIZES=(128 256)
RESULTS_FILE="./result/results.txt"
LATENCIES_FILE="./result/latencies.txt"

# 清空旧的结果文件
echo "Block Size, Mean Latency, Std Dev" > $RESULTS_FILE
echo "Block Size, Latencies" > $LATENCIES_FILE

for block_size in "${BLOCK_SIZES[@]}"
do
    echo "Running experiment with block_size=$block_size..."

    python blocksize_batch_exp.py $block_size $RESULTS_FILE $LATENCIES_FILE

    echo "Completed experiment with block_size=$block_size"
    echo ""
done

echo "All experiments finished! Results saved in $RESULTS_FILE and $LATENCIES_FILE"
