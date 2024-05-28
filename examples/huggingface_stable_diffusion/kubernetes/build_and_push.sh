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
  echo "    export DOCKER_IMAGE_NAME_WITH_TAG=my-registry:5000/stable-diffusion-example:latest"
  exit 1
fi

if [ -z ${FROM_IMAGE_NAME} ]; then
  echo "Provide Docker image that would be used as base image"
  echo "Example:"
  echo "    export FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.04-py3"
  exit 1
fi

BUILD_FROM="${BUILD_FROM:-pypi}"
if [[ ${BUILD_FROM} != "pypi" ]] && [[ ${BUILD_FROM} != "dist" ]]; then
  echo "The BUILD_FROM variable should be equal to 'pypi' or 'dist'"
  echo "Example:"
  echo "    export BUILD_FROM=dist"
  exit 1
fi

set -xe

DOCKER_BUILDKIT=1 docker build -f examples/huggingface_stable_diffusion/kubernetes/Dockerfile \
 -t ${DOCKER_IMAGE_NAME_WITH_TAG} \
 --build-arg FROM_IMAGE_NAME=${FROM_IMAGE_NAME} \
 --build-arg BUILD_FROM=${BUILD_FROM} .
docker push ${DOCKER_IMAGE_NAME_WITH_TAG}