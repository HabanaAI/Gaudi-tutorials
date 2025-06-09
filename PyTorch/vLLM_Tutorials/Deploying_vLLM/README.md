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
|meta-llama/Llama-3.2-11B-Vision-Instruct |1|
|meta-llama/Llama-3.2-90B-Vision-Instruct |4|
## Quick Start
To run these models on your Gaudi machine:

1) First, obtain the Dockerfile and benchmark scripts from the Gaudi-tutorials repository using the command below
```bash
git clone https://github.com/HabanaAI/Gaudi-tutorials
cd Gaudi-tutorials/PyTorch/vLLM_Tutorials/Deploying_vLLM
```

> **IMPORTANT**
>     
> **All build and run steps listed in this document need to be executed on Gaudi Hardware**
>    

2) Depending on the base OS you are running, select the appropriate Dockerfile. The examples in this page are for Ubuntu 24.04
 - Ubuntu 22.04: Dockerfile-1.21.1-ub22-vllm-v0.7.2+Gaudi
 - Ubuntu 24.04: Dockerfile-1.21.1-ub24-vllm-v0.7.2+Gaudi

3) To build the `vllm-v0.7.2-gaudi` image from the Dockerfile, use the command below.
```bash
## Set the next line if you are using a HTTP proxy on your build machine
BUILD_ARGS="--build-arg http_proxy --build-arg https_proxy --build-arg no_proxy"
docker build -f Dockerfile-1.21.1-ub24-vllm-v0.7.2+Gaudi $BUILD_ARGS -t vllm-v0.7.2-gaudi-ub24:1.21.1-16 .
```

4) Set the following variables with appropriate values
 -  -e MODEL= (choose from table above)
 -  -e HF_TOKEN= (Generate a token from https://huggingface.co)

> Note: 
> The Huggingface model file size might be large. Using an external disk to house the Huggingface hub folder is recommended.
> Export the HF_HOME environment variable pointing to the external disk housing the Huggingface hub folder.
> You can do this by adding parameters to the docker run command.  
> Example: "-e HF_HOME=/mnt/huggingface -v /mnt/huggingface:/mnt"

5) Start the vLLM server with a default context (4k for text and 8k for vision models) and default TP as per the table above
```bash
docker run -it --rm \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    --cap-add=sys_nice \
    --ipc=host \
    --runtime=habana \
    -e HF_TOKEN=YOUR_TOKEN_HERE \
    -e HABANA_VISIBLE_DEVICES=all \
    -p 8000:8000 \
    -e MODEL=meta-llama/Llama-3.1-8B-Instruct \
    --name vllm-server \
    vllm-v0.7.2-gaudi-ub24:1.21.1-16
```

6) (Optional) check your vLLM server by running this command in a **separate terminal**
```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct
target=localhost
curl_query="What is DeepLearning?"
payload="{ \"model\": \"${model}\", \"prompt\": \"${curl_query}\", \"max_tokens\": 128, \"temperature\": 0 }"
curl -s --noproxy '*' http://${target}:8000/v1/completions -H 'Content-Type: application/json' -d "$payload"
```

7) Expect to see an output similar to this:
<code>
{"id":"cmpl-694ba4a409444b2a8e2348657a073721","object":"text_completion","created":1747731763,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"text":" Deep learning is a subset of machine learning that uses artificial neural networks to analyze data. It is a type of machine learning that is inspired by the structure and function of the human brain. Deep learning algorithms are designed to learn and improve on their own by analyzing large amounts of data, and they can be used for a wide range of tasks, including image and speech recognition, natural language processing, and predictive modeling.\nDeep learning is a type of machine learning that is particularly well-suited to tasks that involve complex patterns and relationships in data. It is often used in applications such as:\nImage and speech recognition: Deep learning algorithms can be used to","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":6,"total_tokens":134,"completion_tokens":128,"prompt_tokens_details":null}}
</code>
&nbsp; 

8.1) (Optional: For text based models) Run the perftest.sh command in a **separate terminal** for obtaining basic metrics like the example below for Gaudi3:  
```bash
docker exec vllm-server /root/scripts/perftest.sh
```
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

> Note:  
> The perftest.sh script runs with the following defaults:
>   INPUT_TOKENS=2048  
>   OUTPUT_TOKENS=2048  
>   CONCURRENT_REQUESTS=64  

8.2) (Optional: For vision models) Run the perftest_vision.sh command in a **separate terminal** for obtaining basic metrics like the example below for Gaudi3:  
```bash
docker exec vllm-server /root/scripts/perftest_vision.sh
```
<pre>
# meta-llama/Llama-3.2-11B-Vision-Instruct
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  121.53    
Total input tokens:                      31710     
Total generated tokens:                  64000     
Request throughput (req/s):              4.11      
Output token throughput (tok/s):         526.63    
Total Token throughput (tok/s):          787.56    
---------------Time to First Token----------------
Mean TTFT (ms):                          5642.06   
Median TTFT (ms):                        5589.81   
P90 TTFT (ms):                           8825.33   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          74.14     
Median TPOT (ms):                        72.15     
P90 TPOT (ms):                           101.27    
---------------Inter-token Latency----------------
Mean ITL (ms):                           73.56     
Median ITL (ms):                         34.46     
P90 ITL (ms):                            88.77     
==================================================
</pre>

9) Optionally, you can run perftest.sh with custom parameters like so:
```bash
## Usage: docker exec vllm-server /root/scripts/perftest.sh <INPUT_TOKENS> <OUTPUT_TOKENS> <CONCURRENT_REQUESTS>
## Examples:
docker exec vllm-server /root/scripts/perftest.sh 1024 3192
docker exec vllm-server /root/scripts/perftest.sh 1024 3192 100
``` 
&nbsp;

# Running vLLM server with custom parameters
1) The following variables come with defaults but can be overridden with appropriate values
 -  -e TENSOR_PARALLEL_SIZE (Optional, number of cards to use. If not set, a default will be chosen)
 -  -e MAX_MODEL_LEN (Optional, set a length that suits your workload. If not set, a default will be chosen)

2) Example for bringing up a vLLM server with a custom max model length and tensor parallel (TP) size. Proxy variables and volumes added for reference.
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
    -e MODEL=meta-llama/Llama-3.1-70B-Instruct \
    -e TENSOR_PARALLEL_SIZE=8 \
    -e MAX_MODEL_LEN=8192 \
    --name vllm-server \
    vllm-v0.7.2-gaudi-ub24:1.21.1-16
```
3) Example for bringing up two Llama-70B instances with the recommended number of TP/cards. Each instance should have unique values for HABANA_VISIBLE_DEVICES, host port and instance name.
For information on how to set HABANA_VISIBLE_DEVICES for a specific TP size, see [docs.habana.ai - Multiple Tenants](https://docs.habana.ai/en/latest/Orchestration/Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html)
```
CNAME=vllm-v0.7.2-gaudi-ub24:1.21.1-16
HOST_PORT1=8000
docker run -it --rm \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    -e HF_HOME=/mnt/hf_cache \
    -v /mnt/hf_cache:/mnt/hf_cache \
    --cap-add=sys_nice \
    --ipc=host \
    --runtime=habana \
    -e HF_TOKEN=YOUR_TOKEN_HERE \
    -e HABANA_VISIBLE_DEVICES=0,1,2,3 \
    -p $HOST_PORT1:8000 \
    -e MODEL=meta-llama/Llama-3.1-70B-Instruct \
    -e TENSOR_PARALLEL_SIZE=4 \
    -e MAX_MODEL_LEN=8192 \
    --name vllm-server1 \
    ${CNAME}
```

```
## Run in Separate terminal
CNAME=vllm-v0.7.2-gaudi-ub24:1.21.1-16
HOST_PORT2=9222
docker run -it --rm \
    -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
    -e HF_HOME=/mnt/hf_cache \
    -v /mnt/hf_cache:/mnt/hf_cache \
    --cap-add=sys_nice \
    --ipc=host \
    --runtime=habana \
    -e HF_TOKEN=YOUR_TOKEN_HERE \
    -e HABANA_VISIBLE_DEVICES=4,5,6,7
    -p $HOST_PORT2:8000 \
    -e MODEL=meta-llama/Llama-3.1-70B-Instruct \
    -e TENSOR_PARALLEL_SIZE=4 \
    -e MAX_MODEL_LEN=8192 \
    --name vllm-server2 \
    ${CNAME}
```
4) To view vllm-server logs, run this in a separate terminal:
```bash
docker logs -f vllm-server
```
