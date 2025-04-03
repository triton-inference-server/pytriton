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

# Installing PyTriton

This guide shows you how to install PyTriton for your machine learning inference needs.

## Prerequisites

Before installing PyTriton, ensure your system meets these requirements:

- An operating system with glibc >= `2.35` (Ubuntu 22.04+ recommended)
- Python version >= `3.8`
- `pip >= 20.3`

To check your glibc version:
```shell
ldd --version
```

## Quick Installation Guide

The fastest way to install PyTriton is using pip:

```shell
pip install nvidia-pytriton
```

!!! note "Triton Inference Server binaries"
    The Triton Inference Server binaries are automatically installed as part of the PyTriton package.

## Installation Methods

### How to install using system Python

```shell
apt update
apt install -y python3 python3-pip

pip install nvidia-pytriton
```

### How to install using Python virtualenv

```shell
apt update
apt install -y python3 python3-venv python3-pip

# Create and activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

pip install nvidia-pytriton
```

### How to install using uv

```shell
apt update
apt install -y curl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# Create virtual environment (change 3.10 to your desired Python version)
uv venv -p3.10 .venv

# Export the library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(uv run python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

# Install PyTriton in created virtual environment; by default uv use .venv in the current directory, or in the nearest parent directory if no virtual environment is active
uv pip install nvidia-pytriton
```

### How to install using miniconda

```shell
apt update
apt install -y python3 curl

# Download, install and init conda
CONDA_VERSION=latest
TARGET_MACHINE=x86_64
curl "https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-${TARGET_MACHINE}.sh" --output miniconda.sh
bash miniconda.sh
export PATH=~/miniconda3/bin/:$PATH
conda init bash
bash

# Create and activate virtualenv (change 3.10 to your desired Python version)
conda create -c conda-forge -n venv python=3.10
conda activate venv

# Export the library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

pip install nvidia-pytriton
```

### How to install using Docker

If you prefer using Docker:

1. While NVIDIA optimized containers from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers) are recommended for optimal performance, you can use any Docker image with a compatible OS (glibc >= 2.35)
2. For GPU acceleration, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
3. Install PyTriton inside your container using any of the methods above

Example Dockerfile:
```dockerfile
FROM nvcr.io/nvidia/pytorch:25.02-py3
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install nvidia-pytriton
```

## Advanced: Building from Source

You can build PyTriton from source to:
- Access unreleased hotfixes
- Modify the PyTriton code
- Ensure compatibility with various Triton Inference Server versions

For detailed instructions, see the [Building Guide](guides/building.md).

## Reference

### System Requirements

| Requirement | Version |
|-------------|---------|
| Operating System | glibc >= 2.35 (Ubuntu 22.04+, Debian 11+, Rocky Linux 9+, Red Hat UBI 9+) |
| Python | >= 3.8 |
| pip | >= 20.3 |

### Upgrading pip

If you need to upgrade your pip version:

```shell
pip install -U pip
```

### Setting Up LD_LIBRARY_PATH

The Triton Inference Server requires that the `libpython3.*.so` library is accessible. Make sure to set up your `LD_LIBRARY_PATH` environment variable correctly before running PyTriton.
