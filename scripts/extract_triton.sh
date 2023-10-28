#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

export PYTRITON_IMAGE_NAME=$1
export TARGET_DIR=$2
export PLATFORM=$3

rm -rf "${TARGET_DIR}"

export PYTRITON_CONTAINER_ID=$(docker create --platform "$PLATFORM" --rm "${PYTRITON_IMAGE_NAME}" bash -c "sleep 1h")
docker start "${PYTRITON_CONTAINER_ID}"

mkdir -p "${TARGET_DIR}"/backends/python
mkdir -p "${TARGET_DIR}"/caches/local
docker cp "${PYTRITON_CONTAINER_ID}":/opt/tritonserver/bin "${TARGET_DIR}"
docker cp "${PYTRITON_CONTAINER_ID}":/opt/tritonserver/lib "${TARGET_DIR}/external_libs"
docker cp "${PYTRITON_CONTAINER_ID}":/opt/tritonserver/caches/local/libtritoncache_local.so "${TARGET_DIR}/caches/local"
docker cp "${PYTRITON_CONTAINER_ID}":/opt/tritonserver/backends/python/libtriton_python.so "${TARGET_DIR}"/backends/python
docker cp "${PYTRITON_CONTAINER_ID}":/opt/tritonserver/backends/python/triton_python_backend_utils.py "${TARGET_DIR}"/backends/python
docker cp "${PYTRITON_CONTAINER_ID}:/opt/workspace/python_backend_stubs" "${TARGET_DIR}"

mkdir -p "${TARGET_DIR}"/external_libs
function extract_binary_dependencies() {
  BINARY_PATH="${1}"
  export BINARY_PATH
  echo "==== Extracting dependencies of ${BINARY_PATH}"
  DEPS_SYMLINKS=$(docker exec -e BINARY_PATH  "${PYTRITON_CONTAINER_ID}" bash -c 'ldd ${BINARY_PATH} | awk "/=>/ {print \$3}" | sort -u | xargs realpath -s | sed "s/,\$/\n/"')
  for DEP in ${DEPS_SYMLINKS}
  do
      docker cp "${PYTRITON_CONTAINER_ID}:${DEP}" "${TARGET_DIR}/external_libs"
  done
  DEPS_REALPATH=$(docker exec -e BINARY_PATH "${PYTRITON_CONTAINER_ID}" bash -c 'ldd ${BINARY_PATH} | awk "/=>/ {print \$3}" | sort -u | xargs realpath | sed "s/,\$/\n/"')
  for DEP in ${DEPS_REALPATH}
  do
      docker cp "${PYTRITON_CONTAINER_ID}:${DEP}" "${TARGET_DIR}/external_libs"
  done
}

extract_binary_dependencies /opt/tritonserver/bin/tritonserver
extract_binary_dependencies /opt/tritonserver/lib/libtritonserver.so
extract_binary_dependencies /opt/tritonserver/caches/local/libtritoncache_local.so
extract_binary_dependencies /opt/tritonserver/backends/python/libtriton_python.so
extract_binary_dependencies /opt/tritonserver/backends/python/triton_python_backend_stub

docker stop "${PYTRITON_CONTAINER_ID}"
