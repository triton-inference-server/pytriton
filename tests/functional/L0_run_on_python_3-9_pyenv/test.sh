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

apt update
# need git and build dependencies https://github.com/pyenv/pyenv/wiki#suggested-build-environment
apt install -y python3 python3-distutils python-is-python3 git \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9
pyenv virtualenv 3.9 venv
pyenv activate venv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pyenv virtualenv-prefix)/lib

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
