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

SCRIPTS_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

TRITON_DOCKER_IMAGE="${1}"
TRITON_LOCAL_DIR=$(realpath "${2}")
WHEEL_PATH=$(realpath "${3}")
DIST_DIR="$(dirname "${WHEEL_PATH}")"

DOCKER_PLATFORM=${4}
# get arch from DOCKER_PLATFORM
ARCH=$(echo "${DOCKER_PLATFORM}" | cut -d'/' -f2)
WHEEL_PLATFORM=manylinux_2_31_${ARCH}

DOCKER_CONTAINER_ID=$(docker create --rm --pull always --platform ${DOCKER_PLATFORM} -w "${PWD}" "${TRITON_DOCKER_IMAGE}" bash -c "sleep 1h")
docker start "${DOCKER_CONTAINER_ID}"

docker exec "${DOCKER_CONTAINER_ID}" mkdir -p "${DIST_DIR}"
docker exec "${DOCKER_CONTAINER_ID}" mkdir -p "$(dirname "${TRITON_LOCAL_DIR}")"

docker cp "${WHEEL_PATH}" "${DOCKER_CONTAINER_ID}:${WHEEL_PATH}"
docker cp "${TRITON_LOCAL_DIR}" "${DOCKER_CONTAINER_ID}:${TRITON_LOCAL_DIR}"

docker exec "${DOCKER_CONTAINER_ID}" pip install auditwheel==5.3.0 patchelf==0.17.2
docker cp "${SCRIPTS_DIR}/auditwheel_patched.py" "${DOCKER_CONTAINER_ID}:/tmp/"
docker exec "${DOCKER_CONTAINER_ID}" bash -c "LD_LIBRARY_PATH=${TRITON_LOCAL_DIR}/external_libs /tmp/auditwheel_patched.py -vvvv repair --plat ${WHEEL_PLATFORM} ${WHEEL_PATH}"

WHEELHOUSE_DIR="$(dirname "${DIST_DIR}")/wheelhouse"
RESULT_WHEEL_PATH=$(docker exec "${DOCKER_CONTAINER_ID}" bash -c "find ${WHEELHOUSE_DIR} -type f -name *.whl")
docker cp "${DOCKER_CONTAINER_ID}:${RESULT_WHEEL_PATH}" "${DIST_DIR}"

docker stop "${DOCKER_CONTAINER_ID}"
