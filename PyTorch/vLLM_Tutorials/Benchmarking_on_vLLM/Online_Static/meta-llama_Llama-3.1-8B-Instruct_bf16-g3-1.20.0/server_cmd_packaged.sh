#!/bin/sh

export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.05
export VLLM_DECODE_BLOCK_BUCKET_STEP=256
export VLLM_PROMPT_SEQ_BUCKET_MAX=2304
export VLLM_DECODE_BLOCK_BUCKET_MAX=6912
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --block-size 128 \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --download_dir /hf_cache \
        --max-model-len 2304 \
        --gpu-memory-util 0.99 \
        --use-padding-aware-scheduling \
        --max-num-seqs 384 \
        --max-num-prefill-seqs 16 \
        --num_scheduler_steps 16

