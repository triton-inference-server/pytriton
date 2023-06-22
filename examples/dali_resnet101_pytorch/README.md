<!--
Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# ResNet101 PyTorch segmentation example

## Overview

The example presents an inference scenario using DALI and ResNet101.

DALI is a portable, holistic framework for GPU-accelerated data loading and augmentation in deep learning workflows.
DALI supports processing images, videos, and audio data. The key features are: straightforward integration with Triton
Inference Server (using [DALI Backend](https://github.com/triton-inference-server/dali_backend)) and PyTriton,
framework-agnostic processing implementation, batched processing, wide collection of operations and graph-based pipeline
implementation approach.

ResNet101 is a segmentation model. Together with DALI on board they form the following scenario:

1. **Preprocessing** - DALI-based typical ResNet preprocessing. Instead of images the input data is a video.
   Includes GPU decoding (using NVDEC), resize and normalization.
2. **Inference** - the model returns the probabilities of a given class in every pixel.
3. **Postprocessing** - DALI takes the original image and the probabilities and extracts a particular class.

Every step mentioned above is executed by the Triton server. Triton client is used only for reading the test
data from disk and handling the result.

The example consists of following files:

- `server.py` - start the model with Triton Inference Server,
- `client.py` - execute HTTP/gRPC requests to the deployed model,
- `model_inference.py` - ResNet101 inference with PyTorch,

## The sharp bits

Presented scenario is not a straightforward and simple example. There are some sharp bits and we'll try to explain all
of them in this section.

1. **`prefetch_queue_depth` option**. One of main DALI features is the prefetching - loading next iteration when the
   previous one is being processed by the DL model. By default, when the model processes the batch, DALI is preparing
   the next iteration, which is expressed by the default value of the `prefetch_queue_depth = 2` argument. While this is
   really useful for training, it's not so much for inference - we tend to get the data and quickly process it as soon
   as possible. Therefore in most of inference DALI pipeline, the `prefetch_queue_depth = 1`.
1. **`NFCHW -> NCHW` conversion**. When programming DALI pipeline, user does not see the batch dimension - it is hidden.
   DALI assumes, that since every operation will be defined the same way for every sample in a batch, the batch can be
   implicit. Since the input data to the preprocessing pipeline is a video, the preprocessing pipeline returns the data
   in a `NFCHW` layout, where `N` denotes the batch dimension and `F` denotes the frame in a video sequence. Right after
   the preprocessing pipeline, the `NFCHW` layout has to be flattened to `(N*F)CHW` layout to form a batch.
1. **Memory limit**. When using DALI with Triton or PyTriton, there are two ways of decoding the videos:
   using `fn.decoders.video` or using `fn.inputs.video`. The former receives the encoded buffer via `fn.external_source`
   operator and decodes the whole video in one go. On the other hand, the latter is a standalone input to DALI
   pipeline (thus receives the data itself) and decodes only portions of the encoded video, specified
   by `sequence_lenght` operator. This behaviour is required for longer videos, as when they are decoded they can take
   terabytes of RAM. Using `fn.inputs.video` with Triton or PyTriton requires setting DALI model as
   a [`decoupled model`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/decoupled_models.md),
   so that it can generate multiple responses per one request. Since PyTriton does not support decoupled model
   yet, `fn.decoders.video` is used in this example. For more details about the video decoding in DALI please refer to
   the operators documentation: [`fn.decoders.video`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.experimental.decoders.video.html#nvidia.dali.fn.experimental.decoders.video)
   and [`fn.inputs.video`](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.experimental.inputs.video.html).

## Running example

### Prerequisities

This example assumes the following dependencies installed in your system:

1. Docker
2. NumPy
3. OpenCV-Python (optionally, for saving the images to disk)
4. PyTriton (for the `client.py` script)

### Run

To run this example, please follow these steps:

1. Install required dependencies.

2. Run the NVIDIA PyTorch container:

```shell
$ docker run -it --gpus all --shm-size 8gb -v $(pwd):/dali -w /dali --net host nvcr.io/nvidia/pytorch:23.04-py3 bash
```

3. Install PyTriton following the [installation instruction](../../README.md#installation)

4. Inside the container start the Triton server:

```shell
$ python server.py
```

5. In a new terminal window run the Triton client:

```shell
$ python client.py
```

### Extra options

The `client.py` script accepts extra options, listed below:

1. `--dump-images` - If specified, the original and segmented images will be saved to disk (in a `$(cwd)/test_video`
   directory).
2. `--image-paths` - If specified, these paths will be used as the input data for the processing,
   instead of the default sample.

## The result

Original image:

![](images/orig0.jpg)

Segmented image:

![](images/segm0.jpg)