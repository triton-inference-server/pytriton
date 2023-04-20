<!--
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Installation

This page explains how to install the library. We assume that you have a basic understanding of the Python programming language
and are familiar with machine learning models. Using [Docker](https://www.docker.com/) is optional but not required.

## Prerequisites

Before installing the library, ensure that you meet the following requirements:

- An operating system with glibc >= 2.31. Triton Inference Server and PyTriton have only been rigorously tested on Ubuntu 20.04.
  Other supported operating systems include Ubuntu 20.04+, Debian 11+, Rocky Linux 9+, and Red Hat Universal Base Image 9+.
- Python version >= 3.8. If you are using Python 3.9+, see the section "[Installation on Python 3.9+](#installation-on-python-39)" for additional steps.
- pip >= 20.3

The library can be installed in the system environment, a virtual environment, or a [Docker](https://www.docker.com) image.
NVIDIA optimized Docker images for Python frameworks can be obtained from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).
If you want to use the Docker runtime, we recommend that you install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to enable running model inference on NVIDIA GPU.

## Installing using pip

You can install the package from [pypi.org](https://pypi.org/project/nvidia-pytriton/) by running the following command:

```shell
pip install -U nvidia-pytriton
```

**Important**: The Triton Inference Server binary is installed as part of the PyTriton package.

## Installation on Python 3.9+

The Triton Inference Server Python backend is linked to a fixed Python 3.8.
Therefore, if you want to install PyTriton on a different version of Python,
you need to prepare the environment for the Triton Inference Server Python backend.
The environment should be located in the `~/.cache/pytriton/python_backend_interpreter`
directory and should contain the packages `numpy~=1.21` and `pyzmq~=23.0`.

### Using pyenv

```shell
apt update
# need git and build dependencies https://github.com/pyenv/pyenv/wiki\#suggested-build-environment
DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-distutils python-is-python3 git \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# install pyenv
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# compile python 3.8
pyenv install 3.8

# prepare venv
pyenv global 3.8
pip3 install virtualenv
mkdir -p ~/.cache/pytriton/
python -mvenv ~/.cache/pytriton/python_backend_interpreter --copies --clear
source ~/.cache/pytriton/python_backend_interpreter/bin/activate
pip3 install numpy~=1.21 pyzmq~=23.0

# recover system python
deactivate
pyenv global system
```

### Using miniconda

```shell
apt update
apt install -y python3 python3-distutils python-is-python3 curl

CONDA_VERSION=latest
TARGET_MACHINE=x86_64
curl "https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-${TARGET_MACHINE}.sh" --output miniconda.sh

sh miniconda.sh -b -p ~/.cache/conda
rm miniconda.sh
~/.cache/conda/bin/conda create -y -p ~/.cache/pytriton/python_backend_interpreter python=3.8 numpy~=1.21 pyzmq~=23.0
```

## Building binaries from source

The binary package can be built from the source, allowing access to unreleased hotfixes, the ability to modify the PyTriton code, and compatibility with various Triton Inference Server versions, including custom server builds.
For further information on building the PyTriton binary, refer to the [Building](building.md) page.
