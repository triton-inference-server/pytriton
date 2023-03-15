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

# Examples

We provide simple examples how to integrate the PyTorch, TensorFlow2, JAX and simple Python models with Triton Inference
Server using PyTriton. The examples are available
at [GitHub repository](../examples).

## Samples Models Deployment

The list of example models deployments:

- [Add-Sub Python model](../examples/add_sub_python)
- [Add-Sub Python model Jupyter Notebook](../examples/add_sub_notebook)
- [BART PyTorch from HuggingFace](../examples/huggingface_bart_pytorch)
- [BERT JAX from HuggingFace](../examples/huggingface_bert_jax)
- [Identity Python model](../examples/identity_python)
- [Linear RAPIDS/CuPy model](../examples/linear_cupy)
- [Linear RAPIDS/CuPy model Jupyter Notebook](../examples/linear_cupy_notebook)
- [Linear PyTorch model](../examples/identity_python)
- [Multi-Layer TensorFlow2](../examples/mlp_random_tensorflow2)
- [Multi Instance deployment for Linear PyTorch model](../examples/multi_instance_linear_pytorch)
- [Multi Model deployment for Python models](../examples/multiple_models_python)
- [NeMo Megatron GPT model with multi-node support](../examples/nemo_megatron_gpt_multinode)
- [OPT JAX from HuggingFace with multi-node support](../examples/huggingface_opt_multinode_jax)
- [ResNet50 PyTorch from HuggingFace](../examples/huggingface_resnet_pytorch)

## Profiling models

The [Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) can be
used to profile the models served through PyTriton. We have prepared the example of
using Perf Analyzer to profile BART PyTorch. See the example code in
[GitHub repository](../examples/perf_analyzer).

## Kubernetes Deployment

The following examples contains guide how to them on Kubernetes cluster:
- [BART PyTorch from HuggingFace](../examples/huggingface_bart_pytorch)
- [ResNet50 PyTorch from HuggingFace](../examples/huggingface_resnet_pytorch)