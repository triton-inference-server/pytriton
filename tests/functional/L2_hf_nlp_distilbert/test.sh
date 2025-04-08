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
TEST_MODULE="$(dirname "${THIS_SCRIPT_PATH}"|sed 's/\//./g').test"
LOGS_DIR=${LOGS_DIR:-$PWD}
# Create the logs folder if it does not exist
mkdir -p "$LOGS_DIR"

# Set the log path with the date and time
LOG_PATH="$LOGS_DIR/log_$(date '+%Y-%m-%d_%H-%M-%S').txt"

pip install transformers!=4.51.0 datasets --upgrade

python -m"${TEST_MODULE}" \
    --test-time-s 36000 \
    --init-timeout-s 300 \
    --batch-size 16 \
    --sequence-length 128 \
    --seed 20221019 \
    --enable-fault-handler \
    --process-monitoring-interval 600 \
    --log-path "${LOG_PATH}" \
    --compress-logs
