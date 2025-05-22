# VLLM container with multi-model support
This folder contains scripts and configuration files that can be used to build a vLLM container with support for the following models:

# Supported Models:
|Model Name | Recommended TP Size |
|--|--|
|deepseek-ai/DeepSeek-R1-Distill-Llama-70B |4|
|meta-llama/Llama-3.1-70B-Instruct |4|
|meta-llama/Llama-3.1-405B-Instruct |8|
|meta-llama/Llama-3.1-8B-Instruct |1|
|meta-llama/Llama-3.2-1B-Instruct |1|
|meta-llama/Llama-3.2-3B-Instruct |1|
|meta-llama/Llama-3.3-70B-Instruct |4|
|mistralai/Mistral-7B-Instruct-v0.2 |1|
|mistralai/Mixtral-8x22B-Instruct-v0.1 |4|
|mistralai/Mixtral-8x7B-Instruct-v0.1 |2|
|Qwen/Qwen2.5-14B-Instruct |1|
|Qwen/Qwen2.5-32B-Instruct |1|
|Qwen/Qwen2.5-72B-Instruct |4|
|Qwen/Qwen2.5-7B-Instruct |1|

## Quick Start
To run these models on your Gaudi machine:

1) First, obtain the Dockerfile and benchmark scripts from the AICE Internal GitHub repository using the command below
```bash
git clone https://github.com/HabanaAI/Gaudi-tutorials
cd Gaudi-tutorials/PyTorch/vLLM_Tutorials/Deploying_vLLM
```
2) Depending on the base OS you are running, select the appropriate Dockerfile. The examples in this page are for Ubuntu 24.04
 - Ubuntu 22.04: Dockerfile-1.21.0-ub22-vllm-v0.7.2+Gaudi
 - Ubuntu 24.04: Dockerfile-1.21.0-ub24-vllm-v0.7.2+Gaudi

3) To build the `vllm-v0.7.2-gaudi` image from the Dockerfile, use the command below.
```bash
## Set the next line if you are using a HTTP proxy on your build machine
BUILD_ARGS="--build-arg http_proxy --build-arg https_proxy --build-arg no_proxy"
docker build -f Dockerfile-1.21.0-ub24-vllm-v0.7.2+Gaudi $BUILD_ARGS -t vllm-v0.7.2-gaudi-ub24:1.21.0-555 .
```

4) Set the follow variables with appropriate values
 -  -e model= (choose from table above)
 -  -e HF_TOKEN= (Generate a token from https://huggingface.co)

> Note: 
> The Huggingface model file size might be large. Using an external disk to house the Huggingface hub folder is recommended.
> Please export HF_HOME environment variable pointing to the external disk housing Huggingface hub folder.
> In the meantime, export the mount point of the external disk into docker instance.
> ex: "-e HF_HOME=/mnt/huggingface -v /mnt/huggingface:/mnt"

5)  Start the vLLM server with a default context of 4K and default TP from table above
```bash
docker run -it --rm \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    --cap-add=sys_nice \
    --ipc=host \
    --runtime=habana \
    -e HF_TOKEN=YOUR_TOKEN_HERE \
    -e HABANA_VISIBLE_DEVICES=all \
    -p 8000:8000 \
    -e model=meta-llama/Llama-3.1-8B-Instruct \
    --name vllm-server \
    vllm-v0.7.2-gaudi-ub24:1.21.0-555
```

6) (Optional) check your vLLM server by running this command in a **separate terminal**
```bash
model=meta-llama/Llama-3.1-8B-Instruct
target=localhost
curl_query="What is DeepLearning?"
payload="{ \"model\": \"${model}\", \"prompt\": \"${curl_query}\", \"max_tokens\": 128, \"temperature\": 0 }"
curl -s --noproxy '*' http://${target}:8000/v1/completions -H 'Content-Type: application/json' -d "$payload"
```

7) Expect to see an output similar to this:
```json
{"id":"cmpl-694ba4a409444b2a8e2348657a073721","object":"text_completion","created":1747731763,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"text":" Deep learning is a subset of machine learning that uses artificial neural networks to analyze data. It is a type of machine learning that is inspired by the structure and function of the human brain. Deep learning algorithms are designed to learn and improve on their own by analyzing large amounts of data, and they can be used for a wide range of tasks, including image and speech recognition, natural language processing, and predictive modeling.\nDeep learning is a type of machine learning that is particularly well-suited to tasks that involve complex patterns and relationships in data. It is often used in applications such as:\nImage and speech recognition: Deep learning algorithms can be used to","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":134,"completion_tokens":128,"prompt_tokens_details":null}}

```
&nbsp;
 
8) (Optional) Run the `docker exec vllm-server /root/scripts/perftest.sh` command in a **separate terminal** to run a quick benchmark script for obtaining basic metrics like the example below for Gaudi3:
<pre>
# meta-llama/Llama-3.1-8B-Instruct
============ Serving Benchmark Result ============
Successful requests:                     640
Benchmark duration (s):                  359.63
Total input tokens:                      1195639
Total generated tokens:                  1310720
Request throughput (req/s):              1.78
Output token throughput (tok/s):         3644.60
Total Token throughput (tok/s):          6969.21
---------------Time to First Token----------------
Mean TTFT (ms):                          1802.70
Median TTFT (ms):                        1807.23
P90 TTFT (ms):                           3186.42
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.68
Median TPOT (ms):                        16.68
P90 TPOT (ms):                           17.35
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.68
Median ITL (ms):                         15.38
P90 ITL (ms):                            18.49
==================================================
</pre>
<pre>
# meta-llama/Llama-3.1-405B-Instruct
============ Serving Benchmark Result ============
Successful requests:                     640
Benchmark duration (s):                  1514.01
Total input tokens:                      1195639
Total generated tokens:                  1310720
Request throughput (req/s):              0.42
Output token throughput (tok/s):         865.73
Total Token throughput (tok/s):          1655.44
---------------Time to First Token----------------
Mean TTFT (ms):                          15474.97
Median TTFT (ms):                        15500.01
P90 TTFT (ms):                           27013.06
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          66.39
Median TPOT (ms):                        66.38
P90 TPOT (ms):                           71.97
---------------Inter-token Latency----------------
Mean ITL (ms):                           66.39
Median ITL (ms):                         60.01
P90 ITL (ms):                            61.32
==================================================
</pre>
# Running vLLM server with custom parameters
1) The follow variables come with defaults but can be overridden with appropriate values
 -  -e tensor_parallel_size (Optional number of cards to use. If not set, a default will be chosen)
 -  -e max_model_len (Optional, set a length that suits your workload. If not set, a default will be chosen)

2) Example for bringing up a vLLM server with a custom max model length and tensor parallel size. Proxy variables and volumes added for reference.
```bash
docker run -it --rm \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    -e HF_HOME=/mnt/hf_cache \
    -v /mnt/hf_cache:/mnt/hf_cache \
    --cap-add=sys_nice \
    --ipc=host \
    --runtime=habana \
    -e HF_TOKEN=YOUR_TOKEN_HERE \
    -e HABANA_VISIBLE_DEVICES=all \
    -p 8000:8000 \
    -e model=meta-llama/Llama-3.1-70B-Instruct \
    -e tensor_parallel_size=8 \
    -e max_model_len=8192 \
    --name vllm-server \
    vllm-v0.7.2-gaudi-ub24:1.21.0-555
```
