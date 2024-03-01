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

if [[ -z ${NVIDIA_TRITON_SERVER_VERSION} ]]; then
  echo "The NVIDIA_TRITON_SERVER_VERSION variable is not set."
  echo "The script must be executed inside Triton Inference Server - nvcr.io/nvidia/tritonserver:xx.yy-pyt-python-py3"
  exit 1
else
  echo "Found NVIDIA_TRITON_SERVER_VERSION=${NVIDIA_TRITON_SERVER_VERSION}."
fi

export GIT_BRANCH_NAME="r${NVIDIA_TRITON_SERVER_VERSION}"

# Use PYTHON_VERSION from the environment, or default to the specified versions
export PYTHON_VERSIONS=${PYTHON_VERSIONS:-"3.8 3.9 3.10 3.11"}

# Convert string to array
IFS=' '

read -r -a PYTHON_VERSIONS <<< "$PYTHON_VERSIONS"

for version in "${PYTHON_VERSION[@]}"; do
  pyenv install -f ${version}
done

echo "Preparing Python Backend Stubs directory in ${PYTHON_STUBS_DIR}"
PYTHON_STUBS_DIR=${PWD}/python_backend_stubs
if [[ -d "${PYTHON_STUBS_DIR}" ]]; then
  echo "Removing existing directory ${PYTHON_STUBS_DIR}"
  rm -rf "${PYTHON_STUBS_DIR}"
fi

echo "Creating new stubs directory ${PYTHON_STUBS_DIR}"
mkdir "${PYTHON_STUBS_DIR}"

echo "Preparing Python Backend directory"
PYTHON_BACKEND_DIR=${PWD}/python_backend
if [[ -d "${PYTHON_BACKEND_DIR}" ]]; then
  echo "Removing existing Python Backend directory ${PYTHON_BACKEND_DIR}"
  rm -rf "${PYTHON_BACKEND_DIR}"
fi

echo "Cloning Python Backend branch ${GIT_BRANCH_NAME} to ${PYTHON_BACKEND_DIR}."
git clone https://github.com/triton-inference-server/python_backend -b "${GIT_BRANCH_NAME}" ${PYTHON_BACKEND_DIR}

for version in "${PYTHON_VERSION[@]}"; do
  echo "Building Python Backend Stub for Python version ${version}"
  cd "${PYTHON_BACKEND_DIR}"

  echo "Revert the repository state"
  git reset --hard && git clean --force -dfx

  echo "Create build directory for Python version ${version}"
  mkdir build && cd build

  echo "Initialize Python for version ${version}"
  pyenv global "${version}"
  python --version

  echo "Preparing build files for Python version ${version}"
  cmake -DTRITON_ENABLE_GPU=ON \
    -DTRITON_BACKEND_REPO_TAG="${GIT_BRANCH_NAME}" \
    -DTRITON_COMMON_REPO_TAG="${GIT_BRANCH_NAME}" \
    -DTRITON_CORE_REPO_TAG="${GIT_BRANCH_NAME}" \
    -DCMAKE_INSTALL_PREFIX:PATH="$(pwd)/install" ..

  echo "Building triton_python_backend_stub for Python version ${version}"
  make triton-python-backend-stub
  ldd triton_python_backend_stub

  CURRENT_STUB_DIR="${PYTHON_STUBS_DIR}/${version}"
  echo "Moving stub for Python version ${version} to ${CURRENT_STUB_DIR}"
  mkdir "${CURRENT_STUB_DIR}"

  mv triton_python_backend_stub "${CURRENT_STUB_DIR}"/triton_python_backend_stub
done
