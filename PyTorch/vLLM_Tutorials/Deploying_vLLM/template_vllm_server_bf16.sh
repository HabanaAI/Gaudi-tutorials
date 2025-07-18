#!/bin/bash

#@VARS
if [ -n "$PT_HPU_RECIPE_CACHE_CONFIG" ]; then # Checks if using recipe cache
    EXTRA_ARGS+=" --num_gpu_blocks_override $NUM_GPU_BLOCKS_OVERRIDE"
fi

## Start server
python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --block-size $BLOCK_SIZE \
        --dtype $DTYPE \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --download_dir $HF_HOME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --use-padding-aware-scheduling \
        --max-num-seqs $MAX_NUM_SEQS \
        --max-num-prefill-seqs $MAX_NUM_PREFILL_SEQS \
        --num-scheduler-steps 1 \
        --disable-log-requests ${${q}EXTRA_ARGS}
