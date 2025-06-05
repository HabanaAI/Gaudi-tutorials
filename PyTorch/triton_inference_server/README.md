# Triton Inference Server with Gaudi

This document provides instructions on deploying hugging face transformer models on Triton Inference Server (TIS) with Intel® Gaudi® AI accelerator. The overal process involves:

- Create a model repository
- Build the conitaner image
- Lunch a Triton Inference Server
- Query the server

For the purpose of this tutorial, the following models will be deployed:

- [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
- [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)

The document is based on the following:

- [Triton Inference Server Quick Start Guide](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md).
- [Deploying Hugging Face Transformer Models in Triton](https://github.com/triton-inference-server/tutorials/blob/17331012af74eab68ad7c86d8a4ae494272ca4f7/Quick_Deploy/HuggingFaceTransformers/README.md)

> [!NOTE]
> The tutorial is intended to be a reference example only. It may not be tuned for optimal performance.

## Create a Model Repository

The first step is to create a model repository according to the structure detailed in Setting up the model repository and Model repository.
To use the example models here, create a directory called `model_repository` and copy the `qwen2` model  into it:

```bash
mkdir -p model_repository
cp -r qwen2/ model_repository/
```

The `qwen2` folder is organized in the way Triton expects and contains two important files needed to serve models in Triton:

- **config.pbtxt**: This file contians information on the backend use, model input/output details, and custom  parameters to use for execution. More information on the full range of model configuration
properties Triton supports can be found [here](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
- **model.py**: This file includes how Triton should handle the model during the initialization, execution, finalization stages. The Gaudi® AI accelerator specific details are given here. More information regarding python backend usage
can be found [here](https://github.com/triton-inference-server/python_backend#usage).

> [!NOTE]
> To enable a generic model on Gaudi, some modifications are required as detailed in [PyTorch Model Porting](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Model_Porting/index.html#pytorch-user-guide) and [Getting Started with Inference on Intel Gaudi](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Getting_Started_with_Inference.html#inference-using-native-fw).

## Build a Triton Container Image

Since a Triton server is launched within a Docker container, a container image tailored for Intel® Gaudi® AI accelerator is needed. Based on the guidelines detailed in the [Setup_and_Install GitHub repository](https://github.com/HabanaAI/Setup_and_Install/tree/main/dockerfiles), once can build such image:

```bash
git clone https://github.com/HabanaAI/Setup_and_Install/
cd Setup_and_Install/dockerfiles/triton
make build BUILD_OS=ubuntu22.04
```

## Launch the Triton Inference Server

Once the image is built, once can launch the Triton Inference Server in a container with the following command:

```bash
ImageName=triton-installer-2.6.0-ubuntu22.04:1.21.0-555
docker run -it --runtime=habana --name triton_backend --rm -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice \
--net=host --ipc=host -v $PWD/model_repository/:/root/model_repository/ ${ImageName}
```

> [!NOTE]
> In the command above, change the `ImageName` as needed for different driver version.

> [!WARNING]
> For private models, one need to request access to the model and add the access token to the docker command `-e PRIVATE_REPO_TOKEN=<hf_your_huggingface_access_token>`.

Inside the docker, install the necessary prerequisites and lunch the server.

```bash
pip install optimum[habana]
pip uninstall triton -y
PT_HPU_LAZY_MODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 tritonserver --model-repository /root/model_repository/
```

Once the server is successfully  launched, the following outputs will be in the console:

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

A sucessful respond appears as:

```bash
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
< 
* Connection #0 to host localhost left intact
```

Finally install the `clinet.py` prerequisites and query the server

```bash
pip install tritonclient gevent geventhttpclient -q 
```

```py
python client.py --model_name qwen2
```

Part of the respond appears as follow:

```bash
----- Processing: model qwen2 ----- 
-----Stats: model qwen2 ----- 
http client out: <tritonclient.http._requested_output.InferRequestedOutput object at 0x7feb2422b730>
Output: [b"I am working on a project where I need to create a function that takes a list of integers as input and returns a new list containing only the even numbers from the original list. How can I achieve this using Python? You can solve this problem by defining a function that iterates through the given list of integers, checks if each number is even, and if so, adds it to a new list. Here's how you can do it:\n\n```python\ndef get_even_numbers(numbers):\n    even_numbers ="
```

> [!NOTE]
> To use Triton on the client side with Intel® Gaudi®, no specific changes are required. Please refer to [Building a client application](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment#building-a-client-application) section and other details in the [client](https://github.com/triton-inference-server/client) documentation to customize your script.

## Host Multiple Models

Thus far, only one model has been loaded. Triton is capable of loading multiple models as once. To do this, once can add a second model to the model repository:

```bash
cp -r llama2/ model_repository/
```

Again, re-run model launch command above, and wait for the confirmation of successful server launch, i.e.:

```bash
.
.
I0605 22:07:17.252902 1929 server.cc:676]
+--------+---------+--------+
| Model  | Version | Status |
+--------+---------+--------+
| llama2 | 1       | READY  |
| qwen2  | 1       | READY  |
+--------+---------+--------+
```

Next, repeat the steps to query the server with `clinet.py` and update the model list passed via `--model_name`, i.e.

```py
python client.py --model_name qwen2,llama2
```

> [!WARNING]
> The number of available Intel® Gaudi® AI accelerator in your contianer should be (at least) equal to number of deployed models.
