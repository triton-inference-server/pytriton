# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
# noqa: D104
"""Constants for pytriton."""
import pathlib

DEFAULT_HTTP_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRICS_PORT = 8002
TRITON_LOCAL_IP = "127.0.0.1"
TRITON_CONTEXT_FIELD_NAME = "triton_context"
PYTRITON_CACHE_DIR = pathlib.Path.home() / ".cache" / "pytriton"
TRITON_PYTHON_BACKEND_INTERPRETER_DIRNAME = "python_backend_interpreter"
