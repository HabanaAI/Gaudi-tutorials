# Large Model usage with minGPT

This tutorial provides example training scripts to demonstrate different DeepSpeed optimization technologies on the Intel® Gaudi®2 AI accelerator. This tutorial will focus on the memory optimization technologies, including Zero Redundancy Optimizer(ZeRO) and Activation Checkpointing.

## Table of Contents

- [Setup](#setup)
- [Memory Consumptions Under Different DeepSpeed Technologies](#memory-consumptions-under-different-deepspeed-technologies)
- [Use ZeRO to solve the Out-Of-Memory issue](#use-zero-to-solve-the-out-of-memory-issue)

## Example Overview

The PyTorch minGPT example is based on the source code forked from GitHub repository
[minGPT](https://github.com/karpathy/minGPT).

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone https://github.com/HabanaAI/Gaudi-tutorials /path/to/Gaudi-tutorials
cd Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed/
```

### Install Habana DeepSpeed

Please follow the instructions provided in the [Gaudi DeepSpeed User Guide](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide.html) to install the DeepSpeed on Gaudi.

```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
```

## Memory Consumptions Under Different DeepSpeed Technologies

### Preparations

1. Make sure there are available HPU devices. In this tutorial we use 8 HPU devices.
1. To demonstrate the memory, add *--dump-memory* in the command line.
1. To limit the training steps (e.g. 4 steps), add *--steps 4* in the command line.

### Run minGPT with different DeepSpeed technologies

1. Create a big model instead of the default gpt-nano model. This makes the memory variation more obvious during different phases.

Change the model type from *gpt-nano* to *gpt2*

```
--- a/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed/demo_ds.py
+++ b/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed/demo_ds.py
@@ -146,7 +146,7 @@ for a, b in zip(x,y):
 from mingpt.model import GPT

 model_config = GPT.get_default_config()
-model_config.model_type = 'gpt-nano'
+model_config.model_type = 'gpt2'
 model_config.vocab_size = train_dataset.get_vocab_size()
 model_config.block_size = train_dataset.get_block_size()
```

2. Run minGPT with DeepSpeed ZeRO0

```bash
cd /path/to/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config.json --use_hpu --steps 4 --dump-memory
```

The memory consumption on different training phases and the max memory consumption will look like below (in MB):

| step | Before forward (M) | After forward (M) | Before backward (M) | After backward (M) | Before step (M) | After step (M) | Max memeory (M) |
| ---- | ------------------ | ----------------- | ------------------- | ------------------ | --------------- | -------------- | --------------- |
| 0    | 328                | 328               | 328                 | 1726 (max 1735)    | 1726            | 1402           | 1735            |
| 1    | 1726 (max 2700)    | 1726              | 1726                | 2051 (max 2384)    | 2051            | 2051           | **2700**        |
| 2    | 2051               | 2051              | 2051                | 2051 (max 2384)    | 2051            | 1726           | 2384            |
| 3    | 1726               | 1726              | 1726                | 2051 (max 2384)    | 2051            | 1726           | 2384            |

3. Run minGPT with DeepSpeed ZeRO1

```bash
cd /path/to/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero1.json --use_hpu --steps 4 --dump-memory
```

The memory consumption on different training phases and the max memory consumption will look like below (in MB):

| step | Before forward (M) | After forward (M) | Before backward (M) | After backward (M) | Before step (M) | After step (M) | Max memeory (M) |
| ---- | ------------------ | ----------------- | ------------------- | ------------------ | --------------- | -------------- | --------------- |
| 0    | 166                | 166               | 166                 | 830 (max 1056)     | 830             | 835            | **1056**        |
| 1    | 672                | 672               | 672                 | 695 (max 997)      | 695             | 672 (max 857)  | 997             |
| 2    | 672                | 672               | 672                 | 695 (max 997)      | 695             | 672 (max 857)  | 997             |
| 3    | 672                | 672               | 672                 | 695 (max 997)      | 695             | 672 (max 857)  | 997             |

4. Run minGPT with DeepSpeed ZeRO1 and Activation Checkpoiting

```bash
cd /path/to/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero1_ac.json --use_hpu --steps 4 --dump-memory --activation-checkpoint
```

The memory consumption on different training phases and the max memory consumption will look like below (in MB):

| step | Before forward (M) | After forward (M) | Before backward (M) | After backward (M) | Before step (M) | After step (M) | Max memeory (M) |
| ---- | ------------------ | ----------------- | ------------------- | ------------------ | --------------- | -------------- | --------------- |
| 0    | 166                | 166               | 166                 | 581 (max 758)      | 581             | 423 (max 586)  | **758**         |
| 1    | 423                | 423               | 423                 | 446 (max 755)      | 446             | 423 (max 608)  | 755             |
| 2    | 423                | 423               | 423                 | 446 (max 758)      | 446             | 423 (max 608)  | 758             |
| 3    | 423                | 423               | 423                 | 446 (max 758)      | 446             | 423 (max 608)  | 758             |

5. Run minGPT with DeepSpeed ZeRO2

```bash
cd /path/to/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero2.json --use_hpu --steps 4 --dump-memory
```

The memory consumption on different training phases and the max memory consumption will look like below (in MB):

| step | Before forward (M) | After forward (M) | Before backward (M) | After backward (M) | Before step (M) | After step (M) | Max memeory (M) |
| ---- | ------------------ | ----------------- | ------------------- | ------------------ | --------------- | -------------- | --------------- |
| 0    | 166                | 166               | 166                 | 660 (max 993)      | 660             | 682            | **993**         |
| 1    | 520                | 520               | 520                 | 663 (max 935)      | 663             | 523 (max 708)  | 935             |
| 2    | 523                | 523               | 523                 | 568 (max 935)      | 568             | 523 (max 708)  | 935             |
| 3    | 523                | 523               | 523                 | 568 (max 935)      | 568             | 523 (max 708)  | 935             |

5. Run minGPT with DeepSpeed ZeRO3

```bash
cd /path/to/Gaudi-tutorials/PyTorch/Large_Model_DeepSpeed
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero3.json --use_hpu --steps 4 --dump-memory
```

The memory consumption on different training phases and the max memory consumption will look like below (in MB):

| step | Before forward (M) | After forward (M) | Before backward (M) | After backward (M) | Before step (M) | After step (M) | Max memeory (M) |
| ---- | ------------------ | ----------------- | ------------------- | ------------------ | --------------- | -------------- | --------------- |
| 0    | 166                | 166               | 166                 | 660 (max 993)      | 660             | 682            | **993**         |
| 1    | 520                | 520               | 520                 | 663 (max 935)      | 663             | 523 (max 708)  | 935             |
| 2    | 523                | 523               | 523                 | 568 (max 935)      | 568             | 523 (max 708)  | 935             |
| 3    | 523                | 523               | 523                 | 568 (max 935)      | 568             | 523 (max 708)  | 935             |

**Conclusions:**

1. Zero0 (basically the default DDP) takes biggest memory
1. Zero1 & 2 takes less memory than Zero0
1. With Activation Checkpointing, memory decreases even more.

### Use ZeRO to solve the Out-Of-Memory issue on First-Gen Gaudi

Due to the limited memory on HPU device, it may fail to run a big model on HPU with default configuration (e.g. ZeRO0)

1. Create a very big model with minGPT

Change the model type from *gpt-nano* to *gpt2-xl*

```
--- a/PyTorch/examples/DeepSpeed/minGPT/demo_ds.py
+++ b/PyTorch/examples/DeepSpeed/minGPT/demo_ds.py
@@ -146,7 +146,7 @@ for a, b in zip(x,y):
 from mingpt.model import GPT

 model_config = GPT.get_default_config()
-model_config.model_type = 'gpt-nano'
+model_config.model_type = 'gpt2-xl'
 model_config.vocab_size = train_dataset.get_vocab_size()
 model_config.block_size = train_dataset.get_block_size()
```

1. Run minGPT with DeepSpeed ZeRO0

```bash
cd /path/to/Model-References/PyTorch/examples/DeepSpeed/minGPT
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config.json --use_hpu --steps 4 --dump-memory
```

There will be a OOM error from HPU SW stack like below:

```
...
RuntimeError: FATAL ERROR :: MODULE:BRIDGE Exception in Launch thread...
FATAL ERROR :: MODULE:DEVMEM Allocation failed for size::40960000 (39.0625)MB
```

2. Run minGPT with DeepSpeed ZeRO1

```bash
cd /path/to/Model-References/PyTorch/examples/DeepSpeed/minGPT
deepspeed demo_ds.py --deepspeed --deepspeed_config ds_config_zero1.json --use_hpu --steps 4 --dump-memory
```

Via applying ZeRO technology, e.g. ZeRO1, the model can run successfully on HPU.

## Changelog

### 1.11.0

- Updated with ZeRO3 support

### 1.7.1

- Import DeepSpeed and other necessary packages.
- Add new arguments following DeepSpeed's requirements.
- Add other changes to support minGPT with DeepSpeed. Please refer to [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/) for more details.
- Import habana_frameworks.torch.core to load necessary Habana libraries.
- Set dist_backend to "hccl" in deepspeed.init_distributed() to support distributed training on Gaudi.
- Create a HPU device and replace the model_engine.local_rank to this device.
- Add scripts for different ZeRO configurations.
