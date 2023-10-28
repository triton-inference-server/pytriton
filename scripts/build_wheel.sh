#!/usr/bin/env bash
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

set -xe

DOCKER_PLATFORM=${1}

WHEEL_ARCH=$(echo ${DOCKER_PLATFORM} | sed -e 's/^linux\/amd64$/linux_x86_64/g' -e  's/^linux\/arm64$/linux_aarch64/g')
python3 -m build --wheel -C="--build-option=--plat-name" -C="--build-option=${WHEEL_ARCH}" .
python3 -m build --sdist .
