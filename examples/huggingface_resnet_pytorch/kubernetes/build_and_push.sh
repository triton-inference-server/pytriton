#!/bin/bash
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
if [ -z ${DOCKER_IMAGE_NAME_WITH_TAG} ]; then
  echo "Provide Docker image name under to push the created image to your registry"
  echo "Example:"
  echo "    export DOCKER_IMAGE_NAME_WITH_TAG=my-registry:5000/resnet-pytorch-example:latest"
  exit 1
fi

if [ -z ${FROM_IMAGE_NAME} ]; then
  echo "Provide Docker image that would be used as base image"
  echo "Example:"
  echo "    export FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.11-py3"
  exit 1
fi

set -xe

docker build -f examples/huggingface_resnet_pytorch/kubernetes/Dockerfile \
 -t ${DOCKER_IMAGE_NAME_WITH_TAG} \
 --build-arg FROM_IMAGE_NAME=${FROM_IMAGE_NAME} .
docker push ${DOCKER_IMAGE_NAME_WITH_TAG}