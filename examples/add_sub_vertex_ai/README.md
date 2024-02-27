<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

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

# Add-Sub Model with Vertex AI

## Overview

The example presents a simple Add-Sub model launched in Triton Vertex AI mode.

Example consists of following scripts:

- `install.sh` - install additional dependencies
- `server.py` - start the model with Triton Inference Server in Vertex AI mode
- `client.py` - execute HTTP/gRPC requests to the deployed model
- `data.json` - example input data
- `infer_vertext_ai_endpoint.sh` - script to perform inference on Vertex AI endpoint
- `health_vertex_ai_endpoint.sh` - script to check if Vertex AI endpoint is ready

## Quick Start

The step-by-step guide:

1. Install PyTriton following the [installation instruction](../../README.md#installation)
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
6. Run the `client.py` to perform queries on model:

```shell
./client.py
```

7. Run `infer_vertext_ai_endpoint.sh` to perform inference on Vertex AI endpoint:

```shell
./infer_vertext_ai_endpoint.sh
```

8. Run `check_if_ready.sh` to check if Vertex AI endpoint is ready:

```shell
./health_vertex_ai_endpoint.sh
```

