#!/bin/bash

userscalecsv=/workdir/${client_config}

input_tok=$(sed -n '2p' $userscalecsv | cut -d, -f 3)
output_tok=$(sed -n '2p' $userscalecsv | cut -d, -f 4)
con_req=$(sed -n '2p' $userscalecsv | cut -d, -f 5)

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
                 --num-prompts 320 \
                 --max-concurrency $con_req \
                 --metric-percentiles 90 \
2>&1 | tee -a client_1.log
