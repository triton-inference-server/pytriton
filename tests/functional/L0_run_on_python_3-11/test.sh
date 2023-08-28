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

apt update -y
apt install -y software-properties-common

add-apt-repository ppa:deadsnakes/ppa -y

apt install -y python3.11 python3.11-dev libpython3.11 python3.11-distutils python3.11-venv python3-pip python-is-python3 \
   build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
   libffi-dev curl libbz2-dev pkg-config make

python3.11 -m venv /opt/venv
source /opt/venv/bin/activate

if [[ -d "${PYTRITON_DIST_DIR}" ]];then
  export WHEEL_PATH=$(ls ${PYTRITON_DIST_DIR}/*pytriton*.whl)
  pip install "${WHEEL_PATH}[dev]"
else
  pip install nvidia-pytriton
fi

python -m"${TEST_MODULE}" \
    --init-timeout-s 300 \
    --batch-size 32 \
    --seed 2022101915 \
    --verbose
