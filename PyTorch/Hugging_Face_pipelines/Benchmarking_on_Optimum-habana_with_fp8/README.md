# Benchmarking Hugging Face Pipelines with fp8 on Intel&reg; Gaudi&reg; AI Processor
This section contains an example of how to quantize a Hugging Face models from fp32 to fp8 with Intel Gaudi and the Optimum for Intel Gaudi (aka Optimum Habana) library. An easy benchmarking python scripts with related Dockefile is also provided. Hugging Face pipelines take advantage of the Hugging Face Tasks in transformer models, such as text generation, translation, question answering and more. You can read more about Hugging Face pipelines on their main page [here](https://huggingface.co/docs/transformers/main_classes/pipelines)

A jupyter notebook with fp8 instructions and a Benchmark.py for easy benchmarking are provided.
For learning purpose, the jupyter notebook also has instructions on bare metal to get started.
For Gaudi benchmarking purpose, Benchmark.py script will run Llama2 70b, Llama3.1 8b, Llama3.1 70b, and Llama3.1 405b inside docker and generate a report with performance comparsion against published numbers in [Gaudi Model Performance](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html). 

## Requirements
Please make sure to follow [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install Gaudi driver on the system.
### Jupyter Notebook
Please follow [README](https://github.com/intel-ai-tce/Gaudi-tutorials/blob/OH_benchmark/PyTorch/Hugging_Face_pipelines/README.md) to setup environment for Jupyter notebook.
### Benchmark python scripts

To use dockerfile provided for the sample, please follow [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) to setup habana runtime for Docker images.
#### Docker Build
To build the image from the Dockerfile, please follow below command to build the optimum-habana-text-gen image.
```bash
docker build --no-cache -t optimum-habana-text-gen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .
```
#### Docker Run
After docker build, users could follow below command to run and docker instance and users will be in the docker instance under text-generation folder.
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none   --cap-add=ALL --privileged=true  --net=host --ipc=host optimum-habana-text-gen:latest
```
> [!NOTE]
> The Huggingface model file size might be large, so we recommend to use an external disk as Huggingface hub folder. \
> Please export HF_HOME environment variable to your external disk and then export the mount point into docker instance. \
> ex: "-e HF_HOME=/mnt/huggingface -v /mnt:/mnt"

## How to run Benchmark scripts
Benchmark script will run all the models with different input len, output len and batch size and generate a report to compare all published numbers in [Gaudi Model Performance](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html).  
### Gaudi3
Different json file are provided for different Gaudi Software version like 1.19 and 1.20 on Gaudi3.
To do benchmarking on a machine with 8 Gaudi3 cards, just run the below command inside the docker instance. 
```bash
python3 Benchmark.py
```
### Gaudi2
To do benchmarking on a machine with 8 Gaudi2 cards, just run the below command instead inside the docker instance. 
```bash
GAUDI_VER=2 python3 Benchmark.py
```

### Skip Tests
To skip tests for different models, pass related environment and assign its value to 1.  
For example, skip llama3.1 405B model test by following command.  
```bash
skip_llama31_405b=1 python3 Benchmark.py
```
Here are all supported environment variables to pass different tests :  
skip_llama2_70b, skip_llama31_8b, skip_llama31_70b, skip_llama33_70b, skip_llama31_405b

### HTML Report
A html report will be generated under a folder with timestamp, and the html report will look like below the diagram.
> NOTE: There is also a [PerfSpect](https://github.com/intel/PerfSpect) Report for detailed system and Gaudi information.

![image](https://github.com/user-attachments/assets/6fdd36f6-b563-4339-8339-0b55e4916bf8)


