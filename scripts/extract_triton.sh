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

export TRITON_VERSION=$1
export TRITON_CONTAINER="nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-pyt-python-py3"

export TARGET_DIR=$2
rm -rf "${TARGET_DIR}"

export TRITON_CONTAINER_ID=$(docker create --rm "${TRITON_CONTAINER}" bash -c "sleep 1h")
docker start "${TRITON_CONTAINER_ID}"

mkdir -p "${TARGET_DIR}"/backends
docker cp "${TRITON_CONTAINER_ID}":/opt/tritonserver/bin "${TARGET_DIR}"
docker cp "${TRITON_CONTAINER_ID}":/opt/tritonserver/lib "${TARGET_DIR}"
docker cp "${TRITON_CONTAINER_ID}":/opt/tritonserver/backends/python "${TARGET_DIR}"/backends

mkdir -p "${TARGET_DIR}"/external_libs
TRITONSERVER_DEPS_SYMLINKS=$(docker exec  "${TRITON_CONTAINER_ID}" bash -c 'ldd /opt/tritonserver/bin/tritonserver | awk "/=>/ {print \$3}" | sort -u | xargs realpath -s | sed "s/,\$/\n/"')
for TRITONSERVER_DEP in ${TRITONSERVER_DEPS_SYMLINKS}
do
    docker cp "${TRITON_CONTAINER_ID}:${TRITONSERVER_DEP}" "${TARGET_DIR}/external_libs"
done

TRITONSERVER_DEPS=$(docker exec "${TRITON_CONTAINER_ID}" bash -c 'ldd /opt/tritonserver/bin/tritonserver | awk "/=>/ {print \$3}" | sort -u | xargs realpath | sed "s/,\$/\n/"')
for TRITONSERVER_DEP in ${TRITONSERVER_DEPS}
do
    docker cp "${TRITON_CONTAINER_ID}:${TRITONSERVER_DEP}" "${TARGET_DIR}/external_libs"
done

PYTHONBACKEND_DEPS_SYMLINKS=$(docker exec  "${TRITON_CONTAINER_ID}" bash -c 'ldd /opt/tritonserver/backends/python/libtriton_python.so | awk "/=>/ {print \$3}" | sort -u | xargs realpath -s | sed "s/,\$/\n/"')
for PYTHONBACKEND_DEP in ${PYTHONBACKEND_DEPS_SYMLINKS}
do
    docker cp "${TRITON_CONTAINER_ID}:${PYTHONBACKEND_DEP}" "${TARGET_DIR}/external_libs"
done

PYTHONBACKEND_DEPS=$(docker exec "${TRITON_CONTAINER_ID}" bash -c 'ldd /opt/tritonserver/backends/python/libtriton_python.so | awk "/=>/ {print \$3}" | sort -u | xargs realpath | sed "s/,\$/\n/"')
for PYTHONBACKEND_DEP in ${PYTHONBACKEND_DEPS}
do
    docker cp "${TRITON_CONTAINER_ID}:${PYTHONBACKEND_DEP}" "${TARGET_DIR}/external_libs"
done

docker stop "${TRITON_CONTAINER_ID}"
