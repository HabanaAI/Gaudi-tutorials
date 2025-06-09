#!/bin/bash

## Edit the following variables to test for alternate performance scenarios
DATASET=$1
NUM_PROMPTS=$2
CONCURRENT_REQ=$3
DATASET=${DATASET:-"lmarena-ai/vision-arena-bench-v0.1"}
NUM_PROMPTS=${NUM_PROMPTS:-500}
CONCURRENT_REQ=${CONCURRENT_REQ:-64}

cd /root
python3 vllm-fork/benchmarks/benchmark_serving.py \
                 --model $MODEL \
                 --base-url http://localhost:8000 \
                 --backend openai-chat \
                 --endpoint /v1/chat/completions \
                 --dataset-name hf \
                 --dataset-path $DATASET \
                 --hf-split train \
                 --num-prompts $NUM_PROMPTS \
                 --max-concurrency $CONCURRENT_REQ \
                 --metric-percentiles 90 \
2>&1 | tee -a perftest_dataset${DATASET}_prompts${NUM_PROMPTS}_user${CONCURRENT_REQ}.log
