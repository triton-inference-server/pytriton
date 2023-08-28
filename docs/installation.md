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

You should be comfortable with the Python programming language
and know how to work with Machine Learning models. Using [Docker](https://www.docker.com/) is optional and not necessary.

The library can be installed in any of the following ways:

- system environment
- virtualenv
- [Docker](https://www.docker.com/) image

If you opt for using Docker, you can get NVIDIA optimized Docker images for Python frameworks from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).

To run model inference on NVIDIA GPU using the Docker runtime, we recommend that you
install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html), which enables GPU acceleration for containers.

## Prerequisites

Before installing the library, ensure that you meet the following requirements:

- An operating system with glibc >= `2.35`. Triton Inference Server and PyTriton have only been rigorously tested on Ubuntu 22.04.
  Other supported operating systems include Ubuntu Debian 11+, Rocky Linux 9+, and Red Hat Universal Base Image 9+.
  - to check your glibc version, run `ldd --version`
- Python version >= `3.8`
- `pip >= `20.3`
- `libpython3.*.so` available in the operating system (appropriate for Python version).

## Install from `pypi`

You can install the package from [pypi.org](https://pypi.org/project/nvidia-pytriton/) by running the following command:

```shell
pip install -U nvidia-pytriton
```

!!! note "Triton Inference Server binaries"

    The Triton Inference Server binaries are installed as part of the PyTriton package.

## Setting Up Python Environment

The Triton Inference Server is automatically run with your Python interpreter version. To use Triton binary you need
to make sure that `libpython3.*.so` library can be linked during PyTriton start. Install and provide location to
`libpython3.*.so` library in LD_LIBRARY_PATH before you will run PyTriton. Below we presented some options on how
to prepare your Python environment to run PyTriton with common tools.

### Upgrading `pip` version

You need to have `pip` version 20.3 or higher. To upgrade an older version of pip, run this command:

```shell
pip install -U pip
```

### Using system interpreter

When you are running PyTriton on Ubuntu 22.04 install the desired Python interpreter and `libpython3*so.` library.
```shell
# Install necessary packages
apt update -y
apt install -y software-properties-common

# Add repository with various Python versions
add-apt-repository ppa:deadsnakes/ppa -y

# Install Python 3.8
apt install -y python3.8 libpython3.8 python3.8-distutils python3-pip \
     build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
     libffi-dev curl libbz2-dev pkg-config make

# Install library for interpreter
python3.8 -m pip install nvidia-pytriton
```

### Creating virtualenv using `pyenv`

In order to install different version replace the `3.8` with desired Python version in the example below:

```shell
# Install necessary packages
apt update -y
apt install -y python3 python3-distutils python-is-python3 git \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Configure pyenv in current environment
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.8 with shared library support
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.8

# Create and activate virtualenv
pyenv virtualenv 3.8 venv
pyenv activate venv

# export the library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pyenv virtualenv-prefix)/lib

# Install library for interpreter
pip install nvidia-pytriton
```

### Creating virtualenv using `venv`

In order to install different version replace the `3.8` with desired Python version in the example below:

```shell
# Install necessary packages
apt update -y
apt install -y software-properties-common

# Add repository with various Python versions
add-apt-repository ppa:deadsnakes/ppa -y

# Install Python 3.8
apt install -y python3.8 libpython3.8 python3.8-distutils python3.8-venv python3.8-pip python-is-python3 \
     build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
     libffi-dev curl libbz2-dev pkg-config make

# Create and activate virtualenv
python3.8 -m venv /opt/venv
source /opt/venv/bin/activate

# Install library for interpreter
pip install nvidia-pytriton
```

### Creating virtualenv using `miniconda`

In order to install different version replace the `3.8` with desired Python version in the example below:

```shell
# Install necessary packages
apt update -y
apt install -y python3 python3-distutils python-is-python3 curl

# Download miniconda
CONDA_VERSION=latest
TARGET_MACHINE=x86_64
curl "https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-${TARGET_MACHINE}.sh" --output miniconda.sh

# Install miniconda and add to PATH
bash miniconda.sh
export PATH=~/miniconda3/bin/:$PATH

# Initialize bash
conda init bash
bash

# Create and activate virtualenv
conda create -c conda-forge -n venv python=3.8
conda activate venv

# Export the library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Install library for interpreter
pip install nvidia-pytriton
```

## Building binaries from source

The binary package can be built from the source, allowing access to unreleased hotfixes, the ability to modify the PyTriton code, and compatibility with various Triton Inference Server versions, including custom server builds.
For further information on building the PyTriton binary, refer to the [Building](building.md) page.
