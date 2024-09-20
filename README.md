# Intel Gaudi Tutorials

These are the source files for the tutorials on https://developer.habana.ai/.

The tutorials provide step-by-step instructions for PyTorch and PyTorch Lightning on the Intel Gaudi AI Processor, from beginner level to advanced users.

## IMPORTANT: To run these Jupyter Notebooks you will need to follow these steps:
1. Get access to an Intel Gaudi 2 Accelerator card or node.  See the [Get Access](https://developer.habana.ai/get-access/) page on the Developer Website.  Be sure to use port forwarding `ssh -L 8888:localhost:8888 -L 7860:localhost:7860 -L 6006:localhost:6006 ... user@ipaddress` to be able to access the notebook, run the Gradio interface, and use Tensorboard.   Some of the tutorials use all of these features.
2. Run the Intel Gaudi PyTorch Docker image.  Refer to the Docker section of the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#bare-metal-fresh-os-single-click) for more information.  Running the docker image will allow you access to the entire software stack without having to worry about detailed Software installation Steps.
```
docker run -itd --name Gaudi_Docker --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.17.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.2
docker exec -it Gaudi_Docker bash
```
3. Clone this tutorial in your $HOME directory:  `cd ~ && git clone https://www.github.com/habanaai/Gaudi-tutorials`
4. Install Jupyterlab: `python3 -m pip install jupyterlab`
5. Run the Jupyterlab Server, using the same port mapping as the ssh command:  `python3 -m jupyterlab_server --IdentityProvider.token='' --ServerApp.password='' --allow-root --port 8888 --ServerApp.root_dir=$HOME & ` and take the local URL and run that in your browser

The tutorials will cover the following domains and tasks:

### Advanced
- Fine Tuning with LORA and Inference on Hugging Face Llama 2 70B model  
- Full RAG application with TGI-gaudi
- Getting Started with vLLM
- Getting Started with TGI-Gaudi
- RAG application with LocalGPT modified to run on Intel Gaudi
- How to setup and use DeepSpeed for Training Large Language Models 

### Intermediate
- GPU migration Tool
- Debug for Dynamic Shapes
- Running Simple Inference examples with HPU Graph
- Using Hugging Face Pipelines for Inference
- How to use the Gaudi Tensorboard Plug-in or Perfetto for Profiling
- Transformer Reinforcement Learning with Hugging Face
- Running DeepSpeed on PyTorch Lightning with GPT2
- BERT Fine Tuning use PyTorch Lightning

### Getting Started
- Quick Start: Overall review of Model-References, Hugging Face and DeepSpeed
- Inference
- Fine Tuning
- Training a Classifier	Basic PyTorch Tutorial
- Introduction to Lightning	Running simple PyTorch Lightning model, using Gaudi plug-in
