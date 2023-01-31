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

# Changelog

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
