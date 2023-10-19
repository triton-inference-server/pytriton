<!--
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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

# Simple python remote mode example models

## Overview

The example presents a simple python example of remote mode setup (one model is local and two are remote, setup
from separate python scripts).


Example consists of following scripts:

server_remote_mul.py
server_remote_power.py
server_starting_triton.py

- `install.sh` - install additional dependencies
- `server_starting_triton.py` - start the model locally in Triton Inference Server
- `server_remote_mul.py` - start the model remotely in Triton Inference Server
- `server_remote_power.py` - start the other model remotely in Triton Inference Server
- `client.py` - execute HTTP/gRPC requests to the deployed model

## Quick Start

The step-by-step guide:

1. Install NVIDIA PyTriton following the [installation instruction](../../README.md#installation)
2. Install the additional packages using `install.sh`

```shell
./install.sh
```

3. In separate terminals first start triton server using `server_starting_triton.py` and then start
remote models using `server_remote_mul.py` and `server_remote_power.py`:

```shell
./server_starting_triton.py
```

```shell
./server_remote_mul.py
```

```shell
./server_remote_power.py
```

4. Open new terminal tab (ex. `Ctrl + T` on Ubuntu) or window
5. Go to the example directory
6. Run the `client.py` to perform queries on model:

```shell
./client.py
```
