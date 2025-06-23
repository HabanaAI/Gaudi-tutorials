# Qwen 2.5 VL Multi-modal model in vLLM

This document is meant to summarize the multi-modal specific changes that have been introduced in vLLM for Intel Gaudi while enabling Qwen 2.5 VL model. This covers topics such as multi-modal specific environment variables, OOM error handling strategy and other guidelines.

## Installation

The Qwen 2.5-VL model was enabled with [pull request](https://github.com/HabanaAI/vllm-fork/pull/1163/files) in the vllm-fork repository. Use these commands to clone the vllm-fork:

```bash
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork
pip install --upgrade pip
pip install -r requirements-hpu.txt
python setup.py develop
pip install datasets
```

> [!NOTE]
> Currently the images are expected to be 112x112 aligned. To enforce such alignment: `pip install git+https://github.com/malkomes/transformers.git@ac372cd18f836c41f57cdce46094db00019d4280`

## Supported Features

- Text with image(s) and text-only prompts supported.
- Videos as input not currently supported.

## Input pre-requisites

Due to the model's design constraints, the input images must be a multiple of 112 in both width and height dimensions. Please refer to installation section to install the customer transformer that aligns regular sized images for the user automatically.

## Quick Tests

### Server launch

Use the following cmd to launch the server

```bash
VLLM_SKIP_WARMUP=true python -m vllm.entrypoints.openai.api_server --port 8080 --model Qwen/Qwen2.5-VL-3B-Instruct --tensor-parallel-size 1 --max-num-seqs 128 --dtype bfloat16 --gpu-memory-util 0.9 --max-num-batched-tokens 32768 --max-model-len 32768 --block-size 128
```

a successful launch generates the following output:

```bash
.
.
INFO 06-23 16:56:55 [launcher.py:36] Route: /v1/rerank, Methods: POST
INFO 06-23 16:56:55 [launcher.py:36] Route: /v2/rerank, Methods: POST
INFO 06-23 16:56:55 [launcher.py:36] Route: /invocations, Methods: POST
INFO 06-23 16:56:55 [launcher.py:36] Route: /metrics, Methods: GET
INFO:     Started server process [716]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Changes Introduced

### Newer Out of Memory (OOM) Error Handling Strategy

A user may face OOM errors when trying text with image(s) especially in case of multiple and/or large input images.

Two broad scenarios are described with different methods recommended to avoid OOMs:

- If OOM happens right in the beginning of vLLM startup:
  - OOM is likely occurring due to non-availability of free device memory.
  - Decrease `--gpu_memory_utilization` to between `0.3-0.5`
  - This will increase available free device memory that is needed for pre-processing of images.

- If OOM occurs in HPU Graph capturing stage:
  - When using `VLLM_SKIP_WARMUP=true`, OOM is probably occurring due to lack of available HPU graphs capture memory.
  - Increase `VLLM_GRAPH_RESERVED_MEM` (default: 0.1) to `0.4-0.6`.
  - This reduces the memory pre-allocated for KV Cache in favor of capturing HPU Graphs at runtime.

### New Multi-modal Bucketing Environment Parameters

These variables are only applicable for multi-modal models.

| Environment Variable                 | Purpose                                                         | Interaction                                                                                                       | Impact                                                                                                                                                                                                 | Default Value                             |
| ------------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| `VLLM_MULTIMODAL_BUCKETS`            | Defines series of buckets that correlate to image sizes covered | Range is dependent on min_pixels and max_pixels                                                                   | Longer sequence means more coverage for different images and reduced recompilations at runtime but slower warmups. Smaller sequence mean faster warmups but potentially more recompilations at runtime | 1600, 3136, 4096, 6400, 7744, 9216, 12544 |
| `VLLM_GRAPH_MULTIMODAL_PROMPT_RATIO` | Ratio of text vs video memory in the prefill graphs memory pool | Affected by `VLLM_GRAPH_RESERVED_MEM` and `VLLM_GRAPH_PROMPT_RATIO`                                               | Higher values skew the prefill graphs memory in favor of text part of the prompt suitable when fewer images and more text and vice versa                                                               | 0.3                                       |
| `VLLM_FP32_SOFTMAX`                  | Enables FP32 softmax for the LLM model                          | Helps improve the accuray of the langugae model. Setting this to `=1` could have low/marginal performance impact. | `=0`                                                                                                                                                                                                   |
| `VLLM_FP32_SOFTMAX_VISION`           | Enables FP32 softmax for the vision model                       | Helps improve the accuray of the vision model. Setting this to `=1` could have noticable performance impact.      | `=0`                                                                                                                                                                                                   |

#### `VLLM_MULTIMODAL_BUCKETS`

This environment variable is used to define the range of buckets of pre-compiled recipes for multi-modal inputs. Each bucket denotes a number that correlates to the number of pixels in the image i.e. (Width in pixels *Height in pixels)/(14*14). Each bucket needs to be divisible by 64 e.g. 1600, 3200... etc..Ideally the bucket range should cover the smallest and largest expected input image sizes for optimum performance.

As an example, assuming a configuration for minimum pixels in a server deployment is min_pixels=401408. This number is divisible by 112 *112 (112 aligned in both width and height dimensions). The bucket size to cover this image is Bucket = 401408 / (14*14) = 2048, which is also divisible by 64. Thus, the buckets range has to be >= 2048.

#### `VLLM_GRAPH_MULTIMODAL_PROMPT_RATIO`

The environment variable `VLLM_GRAPH_MULTIMODAL_PROMPT_RATIO` determines the ratio of text vs video memory in the memory pool reserved for prefill graphs. The size of prefill graphs memory pool is itself controlled by other variables `VLLM_GRAPH_RESERVED_MEM` and `VLLM_GRAPH_PROMPT_RATIO`.

For example, let us say there is 100GB free device memory available after loading model weights, profiling and applying `--gpu_memory_utilization` flag. Setting `VLLM_GRAPH_RESERVED_MEM=0.1` reserves 10GB as 'usable graph memory' for HPU Graphs and `VLLM_GRAPH_PROMPT_RATIO=0.2` reserves 20% of 'usable graph memory' for prefill graphs (i.e. 20% of 10GB which is 2GB), while 80% is allocated for decode graphs. Now, setting `VLLM_GRAPH_MULTIMODAL_PROMPT_RATIO=0.3` will split this 2GB of prefill graph memory between text (30% of 2GB) and video (70% of 2GB) to accommodate the respective parts of the prompt.
