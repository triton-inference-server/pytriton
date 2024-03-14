# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
ARG FROM_IMAGE_NAME=nvidia/cuda:12.1.1-devel-ubuntu22.04

FROM ${FROM_IMAGE_NAME} as base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl \
        openmpi-bin libopenmpi-dev libb64-0d \
        python3.10 python3-dev python3-pip python-is-python3 pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=off
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

WORKDIR /workspace

# match the version of repository with the version of tensorrt_llm in requirements.txt
RUN git clone https://github.com/NVIDIA/TensorRT-LLM && cd TensorRT-LLM && git checkout v0.8.0 && cd ..

ADD requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ADD ./server.py /workspace/
ADD ./Makefile /workspace/
ADD ./input_data.json /workspace/