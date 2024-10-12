# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:24.09-tf2-py3
ARG BUILD_FROM=pypi

FROM ${FROM_IMAGE_NAME} as base
WORKDIR /opt/app

# Use when build PyTriton from source
FROM base as install-from-dist
COPY dist/*.whl /opt/app
RUN pip install /opt/app/*.whl

# Install from pypi
FROM base as install-from-pypi
RUN pip install -U nvidia-pytriton

FROM install-from-${BUILD_FROM} AS image

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV NCCL_LAUNCH_MODE="PARALLEL"
ENV PYTHONUNBUFFERED=1

RUN apt update -y && apt install -y dnsutils

COPY examples/huggingface_opt_multinode_jax/install.sh /opt/app
RUN /opt/app/install.sh

COPY examples/huggingface_opt_multinode_jax/client.py /opt/app
COPY examples/huggingface_opt_multinode_jax/server.py /opt/app
COPY examples/huggingface_opt_multinode_jax/modeling_flax_opt.py /opt/app
COPY examples/huggingface_opt_multinode_jax/opt_utils.py /opt/app
COPY examples/huggingface_opt_multinode_jax/kubernetes/run.sh /opt/app
COPY examples/huggingface_opt_multinode_jax/kubernetes/health.sh /opt/app

ENTRYPOINT []
