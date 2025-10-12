#!/bin/bash

BLOCK_SIZES=(16 32 64 128 256 512 1024 2048)
BATCH_SIZES=(1 2 4 8 16 32)
RESULTS_FILE="./result/results.txt"
LATENCIES_FILE="./result/latencies.txt"

# 清空旧的结果文件
echo "Block Size, Mean Latency, Std Dev" > $RESULTS_FILE
echo "Block Size, Latencies" > $LATENCIES_FILE

for block_size in "${BLOCK_SIZES[@]}"
do
    for batch_size in "${BATCH_SIZES[@]}"
    do
        echo "Running experiment with block_size=$block_size, batch_size=$batch_size..."

        python blocksize_batch_exp_new.py $block_size $RESULTS_FILE $LATENCIES_FILE $batch_size

        echo "Completed experiment with block_size=$block_size, batch_size=$batch_size"
        echo ""
    done
done

echo "All experiments finished! Results saved in $RESULTS_FILE and $LATENCIES_FILE"
