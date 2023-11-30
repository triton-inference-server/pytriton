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
import os
import pathlib

DEFAULT_HTTP_PORT = 8000
DEFAULT_GRPC_PORT = 8001
DEFAULT_METRICS_PORT = 8002
TRITON_LOCAL_IP = "127.0.0.1"
TRITON_CONTEXT_FIELD_NAME = "triton_context"
TRITON_PYTHON_BACKEND_INTERPRETER_DIRNAME = "python_backend_interpreter"
DEFAULT_TRITON_STARTUP_TIMEOUT_S = 120
CREATE_TRITON_CLIENT_TIMEOUT_S = 10

__DEFAULT_PYTRITON_HOME = os.path.join(os.getenv("XDG_CACHE_HOME", "$HOME/.cache"), "pytriton")
__PYTRITON_HOME = os.path.expanduser(os.path.expandvars(os.getenv("PYTRITON_HOME", __DEFAULT_PYTRITON_HOME)))
PYTRITON_HOME = pathlib.Path(__PYTRITON_HOME).resolve().absolute()
