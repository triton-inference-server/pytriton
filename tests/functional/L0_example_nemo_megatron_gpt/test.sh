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

set -xe

THIS_SCRIPT_DIR="$(realpath --relative-to="${PWD}" "$(dirname "$0")")"
TEST_MODULE="${THIS_SCRIPT_DIR//\//.}.test"

# nemo-toolkit 1.20.0 requires numpy<1.24,>=1.22, numba crashes with never version
pip install "numpy==1.22"

# This is necessary to avoid error in ONNX used my NeMo toolkit for version of protobuf never than 3.7
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -m"${TEST_MODULE}" \
    --timeout-s 300
