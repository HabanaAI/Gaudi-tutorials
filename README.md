# Intel&reg; Gaudi&reg; Tutorials

These are the source files for the tutorials on the [Developer Website](https://www.intel.com/content/www/us/en/developer/platform/gaudi/tutorials.html)

The tutorials provide step-by-step instructions for PyTorch and PyTorch Lightning on the Intel Gaudi AI Processor, from beginner level to advanced users.  These tutorials should be run with a full Intel Gaudi Node of 8 cards. 

## IMPORTANT: To run these Jupyter Notebooks you will need to follow these steps:
1. Get access to an Intel Gaudi 2 Accelerator card or node.  See the [Get Access](https://developer.habana.ai/get-access/) page on the Developer Website.  Be sure to use port forwarding `ssh -L 8888:localhost:8888 -L 7860:localhost:7860 -L 6006:localhost:6006 ... user@ipaddress` to be able to access the notebook, run the Gradio interface, and use Tensorboard.   Some of the tutorials use all of these features.
2. Run the Intel Gaudi PyTorch Docker image.  Refer to the Docker section of the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click) for more information.  Running the docker image will allow you access to the entire software stack without having to worry about detailed Software installation Steps.
```
docker run -itd --name Gaudi_Docker --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.18.0/ubuntu22.04/habanalabs/pytorch-installer-2.4.0
docker exec -it Gaudi_Docker bash
```
3. Clone this tutorial in your $HOME directory:  `cd ~ && git clone https://www.github.com/habanaai/Gaudi-tutorials`
4. Install Jupyterlab: `python3 -m pip install jupyterlab`
5. Run the Jupyterlab Server, using the same port mapping as the ssh command:  `python3 -m jupyterlab_server --IdentityProvider.token='' --ServerApp.password='' --allow-root --port 8888 --ServerApp.root_dir=$HOME & ` and take the local URL and run that in your browser

The tutorials will cover the following domains and tasks:

### Advanced
- [Fine Tuning with LORA and Inference on Hugging Face Llama 2 70B model](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/llama2_fine_tuning_inference/llama2_fine_tuning_inference.ipynb)
- [Full RAG application with TGI-gaudi](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/RAG_Application/RAG-on-Intel-Gaudi.ipynb)
- [Getting Started with vLLM](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Getting_Started_with_vLLM/Getting_Started_with_vLLM.ipynb)
- [Getting Started with TGI-Gaudi](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/TGI_Gaudi_tutorial/TGI_on_Intel_Gaudi.ipynb)
- [RAG application with LocalGPT modified to run on Intel Gaudi](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/localGPT_inference/LocalGPT_Inference.ipynb)
- [How to setup and use DeepSpeed for Training Large Language Models](https://github.com/HabanaAI/Gaudi-tutorials/tree/main/PyTorch/Large_Model_DeepSpeed)

### Intermediate
- [GPU migration Tool](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/GPU_Migration/GPU_Migration.ipynb)
- [Debug for Dynamic Shapes](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Detecting_Dynamic_Shapes/Detecting_Dynamic_Shapes.ipynb)
- [Running Simple Inference examples with HPU Graph](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Inference/Gaudi_inference_ex2.ipynb)
- [Using Hugging Face Pipelines for Inference](https://github.com/HabanaAI/Gaudi-tutorials/tree/main/PyTorch/Hugging_Face_pipelines)
- [How to use the Gaudi Tensorboard Plug-in or Perfetto for Profiling](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Profiling_and_Optimization/Profiler_and_Optimization.ipynb)
- [Transformer Reinforcement Learning with Hugging Face](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Transformer_Reinforcement_Learning/Transformer_Reinforcement_Learning.ipynb)
- [Running DeepSpeed on PyTorch Lightning with GPT2](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/Lightning/DeepSpeed_Lightning/DeepSpeed_Lightning.ipynb)
- [BERT Fine Tuning using PyTorch Lightning](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/Lightning/Finetune%20Transformers/finetune-transformers.ipynb)

### Getting Started
- [Quick Start: Overall review of Model-References, Hugging Face and DeepSpeed](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Intel_Gaudi_Quickstart/Intel_Gaudi_Quick_Start.ipynb)
- [Inference](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Inference/Intel_Gaudi_Inference.ipynb)
- [Fine Tuning](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Fine_Tuning/Intel_Gaudi_Fine_Tuning.ipynb)
- [Training a Classifier Basic PyTorch Tutorial](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Training%20a%20Classifier/cifar10_tutorial.ipynb)
- [Introduction to Lightning, Running a simple PyTorch Lightning model, using Gaudi plug-in](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/Lightning/Introduction/mnist-hello-world.ipynb)