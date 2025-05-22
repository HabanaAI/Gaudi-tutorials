#!/bin/bash

curl_query="What is Deeplearning?"
target="${target:-localhost}"
model="${model:-meta-llama/Llama-3.1-8B-Instruct}"
payload="{ 'model': \"${model}\", 'prompt': \"${curl_query}\", 'max_tokens': 128, 'temperature': 0 }"

curl --noproxy '*' http://${target}:8000/v1/completions -H 'Content-Type: application/json' -d "$payload" 
