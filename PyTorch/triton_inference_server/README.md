# Triton Inference Server with Gaudi

This document provides instructions on deploying hugging face transformer models on Triton Inference Server (TIS) with Intel® Gaudi® AI accelerator. The overal process involves:

- Create a model repository
- Build the container image
- Launch a Triton Inference Server
- Query the server

For the purpose of this tutorial, the following models will be deployed:

- [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
- [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)

The document is based on the following:

- [Triton Inference Server Quick Start Guide](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md).
- [Deploying Hugging Face Transformer Models in Triton](https://github.com/triton-inference-server/tutorials/blob/17331012af74eab68ad7c86d8a4ae494272ca4f7/Quick_Deploy/HuggingFaceTransformers/README.md)

> [!NOTE]
> The tutorial is intended to be a reference example only. It may not be tuned for optimal performance.

## Clone the tutorial

The first step is to clone the repository

```bash
git clone https://github.com/HabanaAI/Gaudi-tutorials.git
cd Gaudi-tutorials/PyTorch/triton_inference_server/
```

## Create a Model Repository

Next, create a model repository according to the structure detailed in [setting up the model repository](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#setting-up-the-model-repository) and [model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md).
To use the example models here, create a directory called `model_repository` and copy the `qwen2` model  into it:

```bash
mkdir -p model_repository
cp -r qwen2/ model_repository/
```

The `qwen2` folder is organized in the way Triton expects and contains two important files needed to serve models in Triton:

- `config.pbtxt`: This file contians information on the backend use, model input/output details, and custom  parameters to use for execution. More information on the full range of model configuration
properties Triton supports can be found [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
- `model.py`: This file includes how Triton should handle the model during the initialization, execution, finalization stages. More information regarding python backend usage
can be found [here](https://github.com/triton-inference-server/python_backend#usage).

A summary of the modifications made to the original triton files to enable Gaudi® AI accelerator is:

- `model.py`: The `habana_args` class contains arguments specific to Gaudi® AI accelerator for proper model initialization and output generatation.
- `utils.py`: This file contains Gaudi® AI accelerator helper functions defining the model initialization similar to ones in optimum-habana examples, i.e [`AutoModelForCausalLM`](https://github.com/huggingface/optimum-habana/blob/df7db95e47be58e39eba3ba73cf7a68f6e81c46a/examples/language-modeling/run_clm.py#L492-L502), quantization, distibuted training, etc.

> [!NOTE]
> To enable a generic model on Gaudi® AI accelerator, some modifications are required as detailed in [PyTorch Model Porting](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/index.html#pytorch-user-guide) and [Getting Started with Inference on Intel Gaudi](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Getting_Started_with_Inference.html#inference-using-native-fw).

## Build a Triton Container Image

Since a Triton server is launched within a Docker container, a container image tailored for Intel® Gaudi® AI accelerator is needed. Based on the guidelines detailed in the [Setup_and_Install GitHub repository](https://github.com/HabanaAI/Setup_and_Install/tree/main/dockerfiles), one can build such image:

```bash
git clone https://github.com/HabanaAI/Setup_and_Install/
cd Setup_and_Install/dockerfiles/triton
make build BUILD_OS=ubuntu22.04
cd ../../..
```

## Launch the Triton Inference Server

Once the image is built, one can launch the Triton Inference Server in a container with the following command:

```bash
ImageName=triton-installer-2.6.0-ubuntu22.04:1.21.0-555
docker run -it --runtime=habana --name triton_backend --rm -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice \
--net=host --ipc=host -v $PWD/model_repository/:/root/model_repository/ ${ImageName}
```

> [!NOTE]
> In the command above, change the `ImageName` for different driver versions, and the `$PWD/model_repository/` if needed.

> [!WARNING]
> To access private models, you must request permission and include the Hugging Face access token in the Docker command: `-e PRIVATE_REPO_TOKEN=<hf_your_huggingface_access_token>`.

Inside the docker, Install the necessary prerequisites and launch the server.

```bash
pip install optimum[habana]
pip uninstall triton -y
PT_HPU_LAZY_MODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 tritonserver --model-repository /root/model_repository/
```

Once the server is successfully launched, the following outputs will be in the console:

```bash
I0605 19:30:49.849066 98 grpc_server.cc:2495] Started GRPCInferenceService at 0.0.0.0:8001
I0605 19:30:49.849242 98 http_server.cc:4619] Started HTTPService at 0.0.0.0:8000
I0605 19:30:49.890324 98 http_server.cc:282] Started Metrics Service at 0.0.0.0:8002
```

> [!WARNING]
> Review the [Intel® Gaudi® PyTorch bridge documentation](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html#execution-modes) to choose the proper exectuion mode for your model.

## Query the Server

Copy over the `client.py` to the Triton container and check the server status.

```bash
docker cp client.py triton_backend:/opt/tritonserver
docker exec -it triton_backend bash
curl -v localhost:8000/v2/health/ready 
```

A successful response appears as follows:

```bash
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
< 
* Connection #0 to host localhost left intact
```

Finally install the `client.py` prerequisites and query the server

```bash
pip install tritonclient gevent geventhttpclient -q 
```

```py
python client.py --model_name qwen2
```

If the execution is successful, part of the response would be:

```bash
----- Processing: model qwen2 ----- 
-----Stats: model qwen2 ----- 
http client out: <tritonclient.http._requested_output.InferRequestedOutput object at 0x7feb2422b730>
Output: [b"I am working on a project where I need to create a function that takes a list of integers as input and returns a new list containing only the even numbers from the original list. How can I achieve this using Python? You can solve this problem by defining a function that iterates through the given list of integers, checks if each number is even, and if so, adds it to a new list. Here's how you can do it:\n\n```python\ndef get_even_numbers(numbers):\n    even_numbers ="
```

> [!NOTE]
> To use Triton on the client side with Intel® Gaudi®, no specific changes are required. Please refer to [Building a client application](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#building-a-client-application) section and other details in the [client](https://github.com/triton-inference-server/client) documentation to customize your script.

## Host Multiple Models

Thus far, only one model has been loaded. Triton is capable of loading multiple models at once. To do so, add a second model to the model repository:

```bash
cp -r llama2/ model_repository/
```

Re-run the server command and wait until the server has completed startup. If both models are started properly, the log should show the following output:

```bash
I0605 22:07:17.252902 1929 server.cc:676]
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| llama2 | 1       | READY  |
| qwen2  | 1       | READY  |
+--------+---------+--------+
```

Next, repeat the steps to query the server with `client.py` and update the model list passed via `--model_name`, i.e.

```py
python client.py --model_name qwen2,llama2
```

> [!WARNING]
> The container will require access to at least one Intel® Gaudi® AI accelerator for each of the models deployed.
