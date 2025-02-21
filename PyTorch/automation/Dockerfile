# Copyright (c) 2023 HabanaLabs, Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#
# Define build arguments for base image configuration
ARG GAUDI_DOCKER_VERSION=1.19.0
ARG UBUNTU_VERSION=22.04
ARG PYTORCH_VERSION=2.5.1

# HabanaLabs Dockerfile PyTorch installer image as base
FROM vault.habana.ai/gaudi-docker/${GAUDI_DOCKER_VERSION}/ubuntu${UBUNTU_VERSION}/habanalabs/pytorch-installer-${PYTORCH_VERSION}:latest

# Define build arguments for component versions
ARG DEEPSPEED_VERSION=1.19.0
ARG OPTIMUM_HABANA_PIP_VERSION=1.15.0
ARG PEFT_VERSION=0.11.1
ARG TGI_GAUDI_VERSION=2.0.5

# Install DeepSpeed from Habana's GitHub repository
RUN pip install git+https://github.com/HabanaAI/DeepSpeed.git@${DEEPSPEED_VERSION}

# Install Optimum Habana from PyPI
RUN pip install optimum-habana==${OPTIMUM_HABANA_PIP_VERSION}

# Install Python dependencies with specified versions
RUN pip install scipy datasets transformers pickleshare ipython peft==${PEFT_VERSION} && \
    pip install -q --upgrade pip

# Install Jupyter using pip
RUN pip install jupyter 

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook when the container starts
ENTRYPOINT ["ssh", "-L", "8888:localhost:8888", "-L", "7860:localhost:7860", "-L", "6006:localhost:6006", "-J", "guest@ipaddress", "user@ipaddress"]
CMD ["sh", "./run_notebooks.sh:"]
