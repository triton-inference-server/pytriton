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
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/nemo:23.06
ARG BUILD_FROM

FROM ${FROM_IMAGE_NAME} as base
WORKDIR /opt/app

COPY examples/nemo_megatron_gpt_multinode/install.sh /opt/app
RUN /opt/app/install.sh

# Use when build PyTriton from source
FROM base as install-from-dist
COPY dist/*.whl /opt/app
RUN pip install /opt/app/*.whl

# Install from pypi
FROM base as install-from-pypi
RUN pip install -U nvidia-pytriton

FROM install-from-${BUILD_FROM} AS image

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/app

RUN apt update -y && apt install -y dnsutils

COPY examples/nemo_megatron_gpt_multinode/client.py /opt/app
COPY examples/nemo_megatron_gpt_multinode/server.py /opt/app
COPY examples/nemo_megatron_gpt_multinode/helpers.py /opt/app
COPY examples/nemo_megatron_gpt_multinode/gpt.py /opt/app
COPY examples/nemo_megatron_gpt_multinode/kubernetes/run.sh /opt/app
COPY examples/nemo_megatron_gpt_multinode/kubernetes/health.sh /opt/app

ENTRYPOINT []
