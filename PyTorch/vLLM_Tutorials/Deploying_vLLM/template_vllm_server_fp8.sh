#!/bin/bash

export PT_HPU_LAZY_MODE=1
unset VLLM_PROMPT_SEQ_BUCKET_MAX

MODEL_BASE=$(echo $MODEL | awk -F '/' '{print $2}')
MODEL_BASE=${${q}MODEL_BASE,,}

if [ -f "./measurement/${${q}MODEL_BASE}/maxabs_quant_$gnum.json" ]; then
	echo "Measurement file found, skipping calibration"
else
	cd /root/scripts/vllm_hpu_ext/calibration/
	echo "UNI_GROUPS is: [$UNI_GROUPS]"

	if [ -n "$UNI_GROUPS" ] && [ "$UNI_GROUPS" != "None" ]; then
        	echo -e 'Calibrate model with unification'
	        ./calibrate_model.sh -m $MODEL -d /root/scripts/dataset-processed.pkl -o /root/scripts/measurement -l 100 -t $MEASUREMENT_TP -g "$UNI_GROUPS"
	else
        	echo -e 'Calibrate model without unification'
	        ./calibrate_model.sh -m $MODEL -d /root/scripts/dataset-processed.pkl -o /root/scripts/measurement -l 100 -t $MEASUREMENT_TP 
	fi
fi

QUANTIZATION="inc"
KV_CACHE_DTYPE=${DTYPE}_${${q}QUANTIZATION}

#@VARS
if [ -n "$PT_HPU_RECIPE_CACHE_CONFIG" ]; then # Checks if using recipe cache
    EXTRA_ARGS+=" --num_gpu_blocks_override $NUM_GPU_BLOCKS_OVERRIDE"
fi

cd /root/scripts
## Start vLLM FP8 server  
QUANT_CONFIG=./measurement/${${q}MODEL_BASE}/maxabs_quant_$gnum.json vllm serve $MODEL \
        --quantization=${${q}QUANTIZATION} \
        --kv_cache_dtype=${${q}KV_CACHE_DTYPE} \
        --tensor-parallel-size=$TENSOR_PARALLEL_SIZE \
        --max-model-len=$MAX_MODEL_LEN \
        --dtype bfloat16 \
        --gpu-memory-util $GPU_MEM_UTILIZATION \
        --use-padding-aware-scheduling \
        --max-num-seqs $MAX_NUM_SEQS \
        --max-num-prefill-seqs $MAX_NUM_PREFILL_SEQS \
        --num_scheduler_steps 1 \
        --weights-load-device cpu \
        --disable-log-requests ${${q}EXTRA_ARGS}
