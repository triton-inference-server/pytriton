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

# Changelog

## 0.3.0 (2023-09-05)

- new: Support for multiple Python versions starting from 3.8+
- new: Added support for [decoupled models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/decoupled_models.md) enabling to support streaming models (alpha state)
- change: Upgraded Triton Inference Server binaries to version 2.36.0. Note that this Triton Inference Server requires glibc 2.35+ or a more recent version.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.36.0](https://github.com/triton-inference-server/server/releases/tag/v2.36.0)


## 0.2.5 (2023-08-24)

- new: Allow to execute multiple PyTriton instances in the same process and/or host
- fix: Invalid flags for Proxy Backend configuration passed to Triton


[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0)

## 0.2.4 (2023-08-10)

- new: Introduced `strict` flag in `Triton.bind` which enables data types and shapes validation of inference callable outputs
  against model config
- new: `AsyncioModelClient` which works in FastAPI and other async frameworks
- fix: `FuturesModelClient` do not raise `gevent.exceptions.InvalidThreadUseError`
- fix: Do not throw TimeoutError if could not connect to server during model verification

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0)

## 0.2.3 (2023-07-21)

- Improved verification of Proxy Backend environment when running under same Python interpreter
- Fixed pytriton.__version__ to represent currently installed version

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0)

## 0.2.2 (2023-07-19)

- Added `inference_timeout_s` parameters to client classes
- Renamed `PyTritonClientUrlParseError` to `PyTritonClientInvalidUrlError`
- `ModelClient` and `FuturesModelClient` methods raise `PyTritonClientClosedError` when used after client is closed
- Pinned tritonclient dependency due to issues with tritonclient >= 2.34 on systems with glibc version lower than 2.34
- Added warning after Triton Server setup and teardown while using too verbose logging level as it may cause a significant performance drop in model inference

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0)

## 0.2.1 (2023-06-28)

- Fixed handling `TritonConfig.cache_directory` option - the directory was always overwritten with the default value.
- Fixed tritonclient dependency - PyTriton need tritonclient supporting http headers and parameters
- Improved shared memory usage to match 64MB limit (default value for Docker, Kubernetes) reducing the initial size for PyTriton Proxy Backend.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0)

## 0.2.0 (2023-05-30)

- Added support for using custom HTTP/gRPC request headers and parameters.

  This change breaks backward compatibility of the inference function signature.
  The undecorated inference function now accepts a list of `Request` instances instead
  of a list of dictionaries. The `Request` class contains data for inputs and parameters
  for combined parameters and headers.

  See [docs/custom_params.md](docs/custom_params.md) for further information

- Added `FuturesModelClient` which enables sending inference requests in a parallel manner.
- Added displaying documentation link after models are loaded.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of [Triton Inference Server](https://github.com/triton-inference-server/) embedded in wheel: [2.33.0](https://github.com/triton-inference-server/server/releases/tag/v2.33.0)

## 0.1.5 (2023-05-12)

- Improved `pytriton.decorators.group_by_values` function
  - Modified the function to avoid calling the inference callable on each individual sample when grouping by string/bytes input
  - Added `pad_fn` argument for easy padding and combining of the inference results
- Fixed Triton binaries search
- Improved Workspace management (remove workspace on shutdown)

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
  - [Triton Inference Server](https://github.com/triton-inference-server/): 2.29.0
  - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.

## 0.1.4 (2023-03-16)

- Add validation of the model name passed to Triton bind method.
- Add monkey patching of `InferenceServerClient.__del__` method to prevent unhandled exceptions.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
  - [Triton Inference Server](https://github.com/triton-inference-server/): 2.29.0
  - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.

## 0.1.3 (2023-02-20)

- Fixed getting model config in `fill_optionals` decorator.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
  - [Triton Inference Server](https://github.com/triton-inference-server/): 2.29.0
  - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.

## 0.1.2 (2023-02-14)

- Fixed wheel build to support installations on operating systems with glibc version 2.31 or higher.
- Updated the documentation on custom builds of the package.
- Change: TritonContext instance is shared across bound models and contains model_configs dictionary.
- Fixed support of binding multiple models that uses methods of the same class.

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
  - [Triton Inference Server](https://github.com/triton-inference-server/): 2.29.0
  - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.

## 0.1.1 (2023-01-31)

- Change: The `@first_value` decorator has been updated with new features:
  - Renamed from `@first_values` to `@first_value`
  - Added a `strict` flag to toggle the checking of equality of values on a single selected input of the request. Default is True
  - Added a `squeeze_single_values` flag to toggle the squeezing of single value ND arrays to scalars. Default is True
- Fix: `@fill_optionals` now supports non-batching models
- Fix: `@first_value` fixed to work with optional inputs
- Fix: `@group_by_values` fixed to work with string inputs
- Fix: `@group_by_values` fixed to work per sample-wise

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
  - [Triton Inference Server](https://github.com/triton-inference-server/): 2.29.0
  - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.

## 0.1.0 (2023-01-12)

- Initial release of PyTriton

[//]: <> (put here on external component update with short summary what change or link to changelog)

- Version of external components used during testing:
  - [Triton Inference Server](https://github.com/triton-inference-server/): 2.29.0
  - Other component versions depend on the used framework and Triton Inference Server containers versions.
    Refer to its [support matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
    for a detailed summary.
