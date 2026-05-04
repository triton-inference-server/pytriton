# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Smoke tests verifying that client-side imports are available on all platforms."""

import sys


def test_model_client_importable():
    from pytriton.client import ModelClient, AsyncioModelClient  # noqa: F401


def test_tensor_importable():
    from pytriton.model_config.tensor import Tensor  # noqa: F401


def test_decorators_importable():
    from pytriton.decorators import batch  # noqa: F401


def test_exceptions_importable():
    from pytriton.client.exceptions import PyTritonClientValueError  # noqa: F401


def test_triton_class_importable():
    # Triton should be importable everywhere; attempting to start it on
    # non-Linux will fail naturally (binary not found), not at import time.
    from pytriton.triton import Triton  # noqa: F401


def test_tritonserver_binary_absent_on_non_linux():
    if sys.platform == "linux":
        return  # full package may have binaries present — skip check
    from pytriton.utils.distribution import get_root_module_path
    tritonserver_dir = get_root_module_path() / "tritonserver"
    assert not tritonserver_dir.exists(), (
        f"tritonserver binary directory should not be present on {sys.platform}, "
        f"but found: {tritonserver_dir}"
    )
