#!/bin/bash

userscalecsv=/workdir/${client_config}

input_tok=$(sed -n '2p' $userscalecsv | cut -d, -f 3)
output_tok=$(sed -n '2p' $userscalecsv | cut -d, -f 4)
con_req=$(sed -n '2p' $userscalecsv | cut -d, -f 5 | tr -d '[:space:]')
num_prompts=$(( con_req*numprompt_mult ))

cd /root
python3 vllm-fork/benchmarks/benchmark_serving.py \
                 --model $model \
                 --base-url http://localhost:8000 \
                 --backend vllm \
                 --dataset-name sonnet \
                 --dataset-path vllm-fork/benchmarks/sonnet.txt \
                 --sonnet-prefix-len 100 \
                 --sonnet-input-len $input_tok \
                 --sonnet-output-len $output_tok \
                 --ignore-eos \
                 --trust-remote-code \
                 --num-prompts $num_prompts \
                 --max-concurrency $con_req \
                 --metric-percentiles 90 \
2>&1 | tee -a perftest_inp${input_tok}_out${output_tok}_user${con_req}.log
