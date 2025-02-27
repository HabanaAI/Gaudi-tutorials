# Copyright (c) 2023 HabanaLabs, Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#
# Define build arguments for base image configuration
ARG GAUDI_DOCKER_VERSION=1.20.0
ARG UBUNTU_VERSION=22.04
ARG PYTORCH_VERSION=2.6.0
ARG TAG=1.20.0-543

# HabanaLabs Dockerfile PyTorch installer image as base
FROM vault.habana.ai/gaudi-docker/${GAUDI_DOCKER_VERSION}/ubuntu${UBUNTU_VERSION}/habanalabs/pytorch-installer-${PYTORCH_VERSION}:${TAG}

# Upgrade pip
RUN pip install -q --upgrade pip

# Install Jupyter using pip
RUN pip install jupyter 
