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

THIS_SCRIPT_PATH="$(realpath --relative-to="$(pwd)" "$0")"
THIS_SCRIPT_DIR="$(dirname "${THIS_SCRIPT_PATH}")"
TEST_MODULE="$(echo "${THIS_SCRIPT_DIR}"|sed 's/\//./g').test"

echo "Installing libb64-dev required by Perf Analyzer"
apt-get update
apt-get install -y libb64-dev
pip install transformers tritonclient[all]==2.28.0

python -m"${TEST_MODULE}" \
    --init-timeout-s 300 \
    --timeout-s 300 \
    --input-data-path "${THIS_SCRIPT_DIR}/input-data.json" \
    --seed 2022101915 \
    --verbose
