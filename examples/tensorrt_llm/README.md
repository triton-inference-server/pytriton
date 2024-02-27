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

# TensorRT-LLM Models Deployment

This example demonstrates how to deploy TensorRT-LLM models using PyTriton.

## Environment Setup

To begin, you need to set up a Docker container containing all the necessary components for working with TensorRT-LLM engines. Follow these steps:

1. Open your terminal and navigate to the root directory of the PyTriton repository.
2. Build the Docker container by executing the following command:

```bash
docker build -t tensorrt_llm -f examples/tensorrt_llm/Dockerfile examples/tensorrt_llm
```

This command creates a Docker image named `tensorrt_llm`, which includes:

- A clone of the TensorRT-LLM repository with a set of [model converters](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples)
- TensorRT-LLM runtime
- PyTriton

All subsequent steps will be performed within this container. Below is an example command to run the container:

```bash
HOST_HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CONTAINER_HF_HOME=/workspace/.cache/huggingface

mkdir -p $HOST_HF_HOME
# Loads HF_TOKEN from the host environment to access private models
docker run \
    --gpus all --rm -it \
    -v ${HOST_HF_HOME}:${CONTAINER_HF_HOME} \
    -e HF_HOME=${CONTAINER_HF_HOME} \
    -e HF_TOKEN \
     tensorrt_llm bash
```

This command initializes the container, allowing access to private models using your Hugging Face token. Authentication is required to download LLama from Hugging Face.

## Preparing TensorRT-LLM Engine

Before using a TensorRT-LLM engine, you must build it from a source model. In this example, we work with two profiles of the Llama 2 7B model:

```bash
make build-engine-llama2-7b-int8
# or
# make build-engine-llama2-7b-basic
```

The `Makefile` contains example profiles. However, these may not be optimal for every setup. Adjust them based on your hardware specifications and refer to the model documentation for optimal performance. If working with different models, you'll need to add new targets to the `Makefile`.

The `server.py` script supports only decoder-only language models [supported by TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file#models), such as Baichuan, BLOOM, Falcon, GPT, GPT-J, GPT-Nemo, GPT-NeoX, LLaMA, LLaMA-v2, Mistral, MPT, OPT, Phi-1.5/Phi-2, Qwen, Replit Code, SantaCoder, and StarCoder. It supports only single-GPU configurations.

## Running the Server

To start the server, execute the following command:

```bash
mpirun --allow-run-as-root -n 1 python3 server.py \
    --engine-dir /workspace/models/meta-llama/Llama-2-7b-hf/engine/llama2-7b-int8-tp1-pp1-float16-bs4-il3072-ol1024 \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --model-name llama2-7b-int8
# or
# make serve-llama2-7b-int8
```

This command initializes the server, making it ready to process requests.

To perform an inference test, open another terminal window and use `curl`:

```bash
curl http://127.0.0.1:8000/v2/models/llama2-7b-int8/generate_stream \
    -H "Content-Type: application/json" \
    -sS \
    -w "\n" \
    -d '{
        "prompt": "San Francisco is a",
        "max_tokens": 128,
        "streaming": true
    }'
```

This command sends a request to your server to generate text based on the prompt "San Francisco is a" limiting the output to a maximum of 128 tokens, and utilizing streaming.

And that's it! You've successfully set up and deployed a TensorRT-LLM model using PyTriton.