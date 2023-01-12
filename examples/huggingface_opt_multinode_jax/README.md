<!--
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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

# Huggingface OPT JAX Multi-node Deployment

This example shows how to easily deploy JAX large language models in a multi-node environment using pytriton. In this tutorial we will be working with [HuggingFace OPT](https://huggingface.co/docs/transformers/model_doc/opt) with up to 530B parameters.

## Quick start

### Environment requirements

Each node must meet following requirements:
- nvidia-docker2 installed
- [proper CUDA and driver versions](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) based on chosen version of framework container

### Docker image

The easiest way of running this example is inside an
[nvcr.io](https://catalog.ngc.nvidia.com/containers) TensorFlow2 container.
Example `Dockerfile` that can be used to run the server:

```Dockerfile
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.11-tf2-py3
FROM ${FROM_IMAGE_NAME}

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV NCCL_LAUNCH_MODE="PARALLEL"

WORKDIR /workdir

COPY install.sh .
RUN ./install.sh
RUN pip install <pytriton package>

COPY . .
```

On each node we have to build the image (or download it from a registry).

```bash
docker build -t jax-llm:latest .
```

### Serve

On each node run:

```bash
docker run --net host --rm --gpus all jax-llm python server.py \
  --server-addr "<master node IP>:<port (e.g. 1234)>" \
  --num-hosts <number of nodes> \
  --host-idx <current node index, master node has index 0> \
  --model-name <model_name> \
  --tp <tensor parallel size>
```

The server expects two inputs:
- `input` - string array of shape (`batch_size`, 1),
- `output_length` - int64 array of shape (`batch-size`, 1).

It returns a sing output:
- `output` - string array of shape (`batch_size`, 1).

To read more about Triton server please visit [Triton docs](https://github.com/triton-inference-server/server#documentation).

### Call

To use our example client run on any machine:

```bash
docker run --net host jax-llm python client.py \
  --server-addr "<master node IP>:8001" \
  --input "<input text>" \
  --output-length <output length>
```

### Model configurations

| model name        | pretrained | source                                   |
|-------------------|------------|------------------------------------------|
| facebook/opt-125m | True       | [HuggingFace](https://huggingface.co/facebook/opt-125m) |
| facebook/opt-350m | True       | [HuggingFace](https://huggingface.co/facebook/opt-350m) |
| facebook/opt-1.3b | True       | [HuggingFace](https://huggingface.co/facebook/opt-1.3b) |
| facebook/opt-2.7b | True       | [HuggingFace](https://huggingface.co/facebook/opt-2.7b) |
| facebook/opt-6.7b | True       | [HuggingFace](https://huggingface.co/facebook/opt-6.7b) |
| facebook/opt-13b | True       | [HuggingFace](https://huggingface.co/facebook/opt-13b) |
| facebook/opt-30b | True       | [HuggingFace](https://huggingface.co/facebook/opt-30b) |
| facebook/opt-66b | True       | [HuggingFace](https://huggingface.co/facebook/opt-66b) |
| random/125M       | False      |                                          |
| random/350M       | False      |                                          |
| random/1.3B       | False      |                                          |
| random/2.7B       | False      |                                          |
| random/5B       | False      |                                          |
| random/6.7B       | False      |                                          |
| random/13B       | False      |                                          |
| random/20B       | False      |                                          |
| random/30B       | False      |                                          |
| random/66B       | False      |                                          |
| random/89B       | False      |                                          |
| random/17B       | False      |                                          |
| random/310B       | False      |                                          |
| random/530B       | False      |                                          |


## Code


To run JAX in multi-GPU multi-node environment we are using [jax.distributed](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html#jax.distributed.initialize) and [jax.experimental.pjit](https://jax.readthedocs.io/en/latest/_modules/jax/experimental/pjit.html) modules. To learn more about using `pjit` and `jax.distrubted` for running multi-node models please visit JAX docs.


Code:

- [server.py](server.py) - this file runs the Triton server (with `--host-idx 0`) or JAX worker (with `--host_idx` greater than 0) on each node. It contains the code that distributes the inputs from the server to the workers.
- [client.py](client.py) - example of a simple client that calls the server with a single sample.
- [opt_utils.py](opt_utils.py) - lower level code used by [server.py](server.py). In this file we define functions that create a sharding strategy, copy model parameters from the cpu into multiple devices and run inference.
- [modeling_flax_opt.py](modeling_flax_opt.py) - slightly modified [HuggingFace file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_flax_opt.py) with OPT model definition. The main difference is that in the HugginFace repository the model is initialized with FP32 weights even when the operations are in FP16. In our file we use FP16 for both storing parameters and performing operations.
