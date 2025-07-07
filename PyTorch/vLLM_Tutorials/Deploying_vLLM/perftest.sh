#!/bin/bash

## Edit the following variables to test for alternate performance scenarios
INPUT_TOK=$1
OUTPUT_TOK=$2
CONCURRENT_REQ=$3
NP_MULTIPLIER=$4
INPUT_TOK=${INPUT_TOK:-2048}
OUTPUT_TOK=${OUTPUT_TOK:-2048}
CONCURRENT_REQ=${CONCURRENT_REQ:-64}
NP_MULTIPLIER=${NP_MULTIPLIER:-10}
NUM_PROMPTS=$(( CONCURRENT_REQ*NP_MULTIPLIER ))
TOTAL_TOK=$(( INPUT_TOK + OUTPUT_TOK ))
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4352}

if [ "$TOTAL_TOK" -gt "$MAX_MODEL_LEN" ]; then
	echo "INPUT_TOK + OUTPUT_TOK > MAX_MODEL_LEN ($MAX_MODEL_LEN)"
	echo "Invalid input combination... exiting!"
        exit -1
fi

cd /root
python3 vllm-fork/benchmarks/benchmark_serving.py \
                 --model $MODEL \
                 --base-url http://localhost:8000 \
                 --backend vllm \
                 --dataset-name sonnet \
                 --dataset-path vllm-fork/benchmarks/sonnet.txt \
                 --sonnet-prefix-len 100 \
                 --sonnet-input-len $INPUT_TOK \
                 --sonnet-output-len $OUTPUT_TOK \
                 --ignore-eos \
                 --trust-remote-code \
                 --num-prompts $NUM_PROMPTS \
                 --max-concurrency $CONCURRENT_REQ \
                 --metric-percentiles 90 \
2>&1 | tee -a perftest_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.log
