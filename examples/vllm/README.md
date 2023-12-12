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

# vLLM example

This example shows how to deploy [vLLM engine](https://vllm.ai/) on PyTriton to generate natural language texts.

## Overview

The example includes the following scripts:

- `install.sh` - installs additional dependencies for vLLM
- `server.py` - starts the model on Triton Inference Server
- `client.sh` - sends a sample request to the model
- `client_streaming.sh` - sends a sample request to the model with streaming support

## Quick Start

The step-by-step guide:

1. Install PyTriton following
   the [installation instruction](../../README.md#installation)
2. Install the additional packages for vLLM using `install.sh`

```shell
./examples/vllm/install.sh
```

3.  In the same terminal, start the model on Triton using `server.py`. You can specify any HF model supported by vLLM.

```shell
./examples/vllm/server.py --model lmsys/vicuna-7b-v1.3
```

vLLM supports [many HF models](https://docs.vllm.ai/en/latest/models/supported_models.html).

4. You can query the model using `curl` with the `generate` endpoint. For example:

```shell
curl http://localhost:8000/v2/models/lmsys_vicuna_7b_v1.3/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 128,
        "temperature": 0
    }'
```

The model will return a text that completes the prompt.

If you want to use streaming support, use the `generate_stream` endpoint instead. For example:

```shell
curl http://localhost:8000/v2/models/lmsys_vicuna_7b_v1.3/generate_stream \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 128,
        "temperature": 0,
        "stream": true
    }'
```

The model will return multiple texts that complete the prompt, one at a time.
