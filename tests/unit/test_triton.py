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
import pathlib
from unittest.mock import PropertyMock

import pytest

from pytriton.exceptions import PyTritonValidationError
from pytriton.triton import Triton, TritonConfig


def test_triton_server_created_with_default_arguments():
    triton = Triton()

    assert triton._triton_server_config["model_repository"] is not None
    assert triton._triton_server_config["backend_directory"] is not None


def test_triton_server_created_with_custom_arguments():
    config = TritonConfig(id="CustomId", model_repository=pathlib.Path("/tmp"), allow_metrics=False)
    triton = Triton(config=config)

    assert triton._triton_server_config["id"] == "CustomId"
    assert triton._triton_server_config["model_repository"] == "/tmp"
    assert triton._triton_server_config["allow_metrics"] is False
    assert triton._triton_server_config["backend_directory"] is not None


def test_triton_server_created_with_custom_arguments_and_env_variables(mocker):
    import os

    updated_environ = {
        **os.environ,
        "PYTRITON_TRITON_CONFIG_GRPC_PORT": "8080",
        "PYTRITON_TRITON_CONFIG_MODEL_REPOSITORY": "/opt",
    }
    mocker.patch("os.environ", new_callable=PropertyMock(return_value=updated_environ))
    config = TritonConfig(id="CustomId", model_repository=pathlib.Path("/tmp"), allow_metrics=False)
    triton = Triton(config=config)

    assert triton._triton_server_config["id"] == "CustomId"
    assert triton._triton_server_config["model_repository"] == "/tmp"
    assert triton._triton_server_config["grpc_port"] == 8080
    assert triton._triton_server_config["allow_metrics"] is False
    assert triton._triton_server_config["backend_directory"] is not None


def test_triton_bind_model_name_verification():
    triton = Triton()
    triton.bind("AB-cd_90.1", lambda: None, [], [])

    with pytest.raises(
        PyTritonValidationError,
        match="Model name can only contain alphanumeric characters, dots, underscores and dashes",
    ):
        triton.bind("AB#cd/90/1", lambda: None, [], [])

    with pytest.raises(PyTritonValidationError, match="Model name cannot be empty"):
        triton.bind("", lambda: None, [], [])
