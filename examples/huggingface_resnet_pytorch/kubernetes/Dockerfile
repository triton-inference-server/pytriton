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
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.11-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /opt/app

COPY examples/huggingface_resnet_pytorch/install.sh /opt/app
RUN /opt/app/install.sh

# Install from pypi
RUN pip install pytriton

# Use when build PyTriton from source
#COPY dist/*.whl /opt/app
#RUN pip install /opt/app/*.whl


COPY examples/huggingface_resnet_pytorch/client.py /opt/app
COPY examples/huggingface_resnet_pytorch/server.py /opt/app

ENTRYPOINT []