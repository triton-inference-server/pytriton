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

THIS_SCRIPT_DIR="$(realpath --relative-to="${PWD}" "$(dirname "$0")")"

pip install pytest-timeout numpy
pytest -svvv \
    --log-cli-level=DEBUG \
    --log-cli-format='%(asctime)s [%(levelname)s] [%(process)d:%(thread)d] %(name)s:%(lineno)d:  %(message)s' \
    --timeout=60 \
    ${THIS_SCRIPT_DIR}/test_pytest.py