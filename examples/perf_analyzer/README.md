<!--
Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Using Perf Analyzer for profiling HuggingFace BART model

## Overview

The example presents profiling of HuggingFace BART model using [Perf
Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)

Example consists of following scripts:

- `install.sh` - install additional packages and libraries required to run the example
- `server.py` - start the model with Triton Inference Server
- `client.py` - execute HTTP/gRPC requests to the deployed model

## Requirements

The example requires the `torch` package. It can be installed in your current environment using pip:

```shell
pip install torch
```

Or you can use NVIDIA PyTorch container:

```shell
docker run -it --gpus 1 --shm-size 8gb -v {repository_path}:{repository_path} -w {repository_path} nvcr.io/nvidia/pytorch:24.05-py3 bash
```

If you select to use container we recommend to install
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Quick Start

The step-by-step guide:

1. Install PyTriton following
   the [installation instruction](../../README.md#installation)
2. Install the additional packages using `install.sh`

```shell
./install.sh
```

3. In current terminal start the model on Triton using `server.py`

```shell
./server.py
```

4. Open new terminal tab (ex. `Ctrl + T` on Ubuntu) or window
5. Go to the example directory
6. Run the `client.sh` to perform queries on model:

```shell
./client.sh
```

