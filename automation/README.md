Gaudi-tutorials/automation

## Overview

This automation directory contains a Dockerfile, docker_build.sh and run_test.sh files that sets up a PyTorch environment with Jupyter support for HabanaLabs Gaudi AI accelerators. The image is based on Habana's official PyTorch installer image and is designed for use with Ubuntu.

### Prerequisites

Before building the Docker image, ensure you have the following installed:

1. Docker
2. Access to the HabanaLabs container registry

### Docker Build Arguments
The Dockerfile allows customization through the following build arguments:

GAUDI_DOCKER_VERSION (default: 1.20.0): Specifies the version of the Habana Gaudi Docker image.
UBUNTU_VERSION (default: 22.04): Specifies the Ubuntu version.
PYTORCH_VERSION (default: 2.6.0): Specifies the PyTorch version.
TAG (default: 1.20.0-543): Specifies the image tag.


### Building the Image
To build the Docker image, run the following command in your terminal:

```
./docker_build.sh
```

Once the image is built, you can run a run_test.sh to validate the Jupyter Notebook (.ipynb) files found within a specified base directory. It converts the notebooks to Python scripts, replaces a placeholder token, executes them in a Docker container, and logs the results.

### Execution Steps

Ensure Docker is running.
Run the script with the desired parameters:

```
./run_test.sh -d /path/to/directory -t my_token
```

The script processes .ipynb files and logs execution details.
Check the output and log files for execution results.

### Options:

-d base_directory : Specifies the directory where the .ipynb files are located. Default is the current directory (.).

-t token_key : Hugging Face Token key to be used in processing.

### How It Works

1. Parses command-line arguments.
2. Checks if the specified base directory exists.
3. Searches for subdirectories containing .ipynb files.
4. Converts each notebook to a Python script using jupyter nbconvert.
5. Runs the generated Python script inside a Docker container using the sapdai/gaudi-tutorial-env image.
6. Logs the execution status and cleans up temporary files and containers.

### Cleanup

The script automatically removes temporary shell scripts (.sh), Converted Python scripts (.py) and Docker containers created used for execution.

### Tasks Completed

The following tutorials have been successfully verified through automation.
1. Detecting_Dynamic_Shapes
2. Fine_Tuning
3. Inference
4. Intel_Gaudi_Quickstart
5. Profiling_and_Optimization
6. Single_card_tutorials/Intel_Gaudi_Quick_Start_single
7. Transformer_Reinforcement_Learning

### Work in progress

Work is in progress for the remaining tutorials.

### Note
The following items are not functioning within the automation process, and the tutorials require modification accordingly.
1. vLLM_Tutorials
2. TGI_Gaudi_tutorial
3. RAG_Application
