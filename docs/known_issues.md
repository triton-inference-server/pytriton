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

# Known Issues and Limitations

## Feature Limitations

- **Feature Disparity**: PyTriton does not provide feature parity with [Triton Inference Server](https://github.com/triton-inference-server/server), particularly in supporting a user model store.
- **Concurrent Execution**: Running multiple scripts hosting PyTriton on the same machine or container is not supported.

## Stability and Performance Issues

- **NCCL Deadlocks**: When using the NCCL communication library, deadlocks may occur if multiple Inference Callables are triggered concurrently (such as when deploying multiple instances of the same model or multiple models within a single server script). For more details, see the [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently).
- **Logging Performance Impact**: Enabling verbose logging can significantly reduce model inference performance.

## Client Limitations

- **GRPC Timeout Support**: The GRPC ModelClient does not support timeouts for model configuration and model metadata requests due to limitations in the underlying tritonclient library.
- **HTTP Timeout Handling**: The HTTP ModelClient may not correctly respect specified timeouts for model initialization and inference requests, especially for timeouts under 1 second. This is caused by the underlying HTTP protocol implementation.

## Benign Issues

- **False Error Messages**: Triton logs may contain a false negative error: `Failed to set config modification time: model_config_content_name_ is empty`. This message can be safely ignored.
