# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM vault.habana.ai/gaudi-docker/1.21.2/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:1.21.2-76

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    npm
RUN npm install n -g && \
    n latest

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --upgrade-strategy eager
RUN python3 -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.20.0
RUN mkdir -p /workspace
WORKDIR /workspace
RUN git clone -b v1.18.0 https://github.com/huggingface/optimum-habana && \
    cd optimum-habana && \
    python3 setup.py install

WORKDIR /workspace/optimum-habana/examples/text-generation
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r requirements_lm_eval.txt
COPY . .
COPY Gaudi_1-21.json Gaudi.json
COPY HQT_1-20.zip HQT.zip
RUN python3 -m pip install -r requirements_bm.txt
