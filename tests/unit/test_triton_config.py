# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import pytest

from pytriton.server.python_backend_config import PythonBackendConfig
from pytriton.server.triton_server_config import TritonServerConfig
from pytriton.triton import INITIAL_BACKEND_SHM_SIZE, TritonConfig


def test_triton_config_raise_with_positional_args():
    with pytest.raises(TypeError, match="TritonConfig initialization can't be used with positional arguments"):
        TritonConfig("CustomId", "/tmp", False)


def test_triton_config_serialization_handles_lists():
    config = TritonConfig(cache_config=["local,size=1048576", "redis,size=10485760"])

    triton_server_config = TritonServerConfig()
    for name, value in config.to_dict().items():
        if name not in TritonServerConfig.allowed_keys() or value is None:
            continue

        triton_server_config[name] = value

    cli = triton_server_config.to_cli_string()
    assert "--cache-config=local,size=1048576" in cli
    assert "--cache-config=redis,size=1048576" in cli


def test_triton_config_serialize_backend_configuration():
    config = PythonBackendConfig()
    config["shm_default_byte_size"] = INITIAL_BACKEND_SHM_SIZE
    cli_backend = config.to_list_args()
    assert [f"python,shm-default-byte-size={INITIAL_BACKEND_SHM_SIZE}"] == cli_backend

    triton_server_config = TritonServerConfig()
    triton_server_config["backend-config"] = cli_backend
    cli = triton_server_config.to_cli_string()

    assert f"--backend-config=python,shm-default-byte-size={INITIAL_BACKEND_SHM_SIZE}" == cli
