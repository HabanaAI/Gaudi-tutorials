#!/usr/bin/env bash

export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=100000
export VLLM_PROMPT_BS_BUCKET_MAX=16
export VLLM_DECODE_BS_BUCKET_MAX=128
export VLLM_DECODE_BLOCK_BUCKET_MIN=2048
export VLLM_DECODE_BLOCK_BUCKET_MAX=4096
export VLLM_PROMPT_SEQ_BUCKET_MAX=2048
export VLLM_PROMPT_SEQ_BUCKET_MIN=2048
export VLLM_SKIP_WARMUP="true"

MODEL_NAME=$1

python -m vllm.entrypoints.openai.api_server --port 8080 --model $MODEL_NAME \
--tensor-parallel-size 8 --max-num-seqs 128 --disable-log-requests --dtype bfloat16 \
--block-size 128 --gpu-memory-util 0.9 --num-lookahead-slots 1 --use-v2-block-manager \
--max-num-batched-tokens 32768 --max-model-len 4096 --quantization inc --kv-cache-dtype fp8_inc \
--weights-load-device cpu 2>&1 | tee server_TP8_fp8.log