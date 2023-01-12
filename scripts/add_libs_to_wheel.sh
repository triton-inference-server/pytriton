#!/usr/bin/env bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

set -x
set -e

export TRITON_VERSION="${1}"
export TRITON_CONTAINER="nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-pyt-python-py3"

export TRITONSERVER_LOCAL_DIR=$(realpath "${2}")
export WHEEL_PATH=$(realpath "${3}")
export DIST_DIR="$(dirname "${WHEEL_PATH}")"
export WHEELHOUSE_DIR="$(dirname "${DIST_DIR}")/wheelhouse"

TRITON_CONTAINER_ID=$(docker create --rm -w "${PWD}" "${TRITON_CONTAINER}" bash -c "sleep 1h")
docker start "${TRITON_CONTAINER_ID}"

docker exec "${TRITON_CONTAINER_ID}" mkdir -p "${DIST_DIR}"
docker cp "${WHEEL_PATH}" "${TRITON_CONTAINER_ID}:${WHEEL_PATH}"
docker exec "${TRITON_CONTAINER_ID}" mkdir -p "$(dirname "${TRITONSERVER_LOCAL_DIR}")"
docker cp "${TRITONSERVER_LOCAL_DIR}" "${TRITON_CONTAINER_ID}:${TRITONSERVER_LOCAL_DIR}"

docker exec "${TRITON_CONTAINER_ID}" pip install auditwheel patchelf
docker exec "${TRITON_CONTAINER_ID}" bash -c "LD_LIBRARY_PATH=${TRITONSERVER_LOCAL_DIR}/lib auditwheel repair --plat linux_x86_64 ${WHEEL_PATH}"
RESULT_WHEEL_PATH=$(docker exec "${TRITON_CONTAINER_ID}" bash -c "find ${WHEELHOUSE_DIR} -type f -name *.whl")
docker cp "${TRITON_CONTAINER_ID}:${RESULT_WHEEL_PATH}" "${DIST_DIR}"

docker stop "${TRITON_CONTAINER_ID}"