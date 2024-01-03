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
import json
import logging
from typing import Union

import numpy as np
import wrapt
from google.protobuf import json_format  # pytype: disable=pyi-error
from tritonclient.grpc import InferenceServerClient as SyncGrpcInferenceServerClient
from tritonclient.grpc import model_config_pb2, service_pb2
from tritonclient.http import InferenceServerClient as SyncHttpInferenceServerClient
from tritonclient.http.aio import InferenceServerClient as AsyncioHttpInferenceServerClient

from pytriton.model_config import DeviceKind
from pytriton.model_config.generator import ModelConfigGenerator
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig

_LOGGER = logging.getLogger(__name__)


ADD_SUB_WITH_BATCHING_MODEL_CONFIG = TritonModelConfig(
    model_name="AddSub",
    model_version=1,
    max_batch_size=16,
    instance_group={DeviceKind.KIND_CPU: 1},
    inputs=[
        TensorSpec(name="a", shape=(-1, 1), dtype=np.float32),
        TensorSpec(name="b", shape=(-1, 1), dtype=np.float32),
    ],
    outputs=[
        TensorSpec(name="add", shape=(-1, 1), dtype=np.float32),
        TensorSpec(name="sub", shape=(-1, 1), dtype=np.float32),
    ],
    decoupled=False,
    backend_parameters={"workspace-path": "dummy/path"},
)

ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG = TritonModelConfig(
    model_name="AddSub",
    model_version=1,
    batching=False,
    instance_group={DeviceKind.KIND_CPU: 1},
    inputs=[
        TensorSpec(name="a", shape=(1,), dtype=np.float32),
        TensorSpec(name="b", shape=(1,), dtype=np.float32),
    ],
    outputs=[
        TensorSpec(name="add", shape=(1,), dtype=np.float32),
        TensorSpec(name="sub", shape=(1,), dtype=np.float32),
    ],
    decoupled=False,
    backend_parameters={"workspace-path": "dummy/path"},
)

ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG_DECOUPLED = TritonModelConfig(
    model_name="AddSub",
    model_version=1,
    batching=False,
    instance_group={DeviceKind.KIND_CPU: 1},
    inputs=[
        TensorSpec(name="a", shape=(1,), dtype=np.float32),
        TensorSpec(name="b", shape=(1,), dtype=np.float32),
    ],
    outputs=[
        TensorSpec(name="add", shape=(1,), dtype=np.float32),
        TensorSpec(name="sub", shape=(1,), dtype=np.float32),
    ],
    decoupled=True,
    backend_parameters={"workspace-path": "dummy/path"},
)


GRPC_LOCALHOST_URL = "grpc://localhost:8001"
HTTP_LOCALHOST_URL_NO_SCHEME = "localhost:8000"
HTTP_LOCALHOST_URL = f"http://{HTTP_LOCALHOST_URL_NO_SCHEME}"

EXPECTED_KWARGS_DEFAULT = {
    "model_name": ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
    "model_version": "",
    "request_id": "0",
    "parameters": None,
    "headers": None,
}

_TritonClientType = Union[
    AsyncioHttpInferenceServerClient, SyncHttpInferenceServerClient, SyncGrpcInferenceServerClient
]
_HttpTritonClientType = Union[AsyncioHttpInferenceServerClient, SyncHttpInferenceServerClient]
_GrpcTritonClientType = SyncGrpcInferenceServerClient


def patch_client__server_up_and_ready(
    mocker, base_triton_client: _TritonClientType, ready_server: bool = True, live_server: bool = True
):
    mocker.patch.object(base_triton_client, base_triton_client.is_server_ready.__name__).return_value = ready_server
    mocker.patch.object(base_triton_client, base_triton_client.is_server_live.__name__).return_value = live_server


def patch_http_client__model_up_and_ready(
    mocker,
    model_config: TritonModelConfig,
    base_triton_client: _HttpTritonClientType,
    ready: bool = True,
):
    mocker.patch.object(base_triton_client, base_triton_client.is_model_ready.__name__).return_value = ready

    model_config_dict = ModelConfigGenerator(model_config).get_config()
    mock_get_model_config = mocker.patch.object(base_triton_client, base_triton_client.get_model_config.__name__)
    mock_get_model_config.return_value = model_config_dict


def patch_grpc_client__model_up_and_ready(
    mocker,
    model_config: TritonModelConfig,
    base_triton_client: _GrpcTritonClientType,
    ready: bool = True,
):
    def new_is_model_ready(model_name, model_version="", headers=None, parameters=None):
        return (
            ready
            and model_name == model_config.model_name
            and (model_version == "" or model_version == str(model_config.model_version))
        )

    mocker.patch.object(base_triton_client, base_triton_client.is_model_ready.__name__, side_effect=new_is_model_ready)

    model_config_dict = ModelConfigGenerator(model_config).get_config()
    model_config_protobuf = json_format.ParseDict(model_config_dict, model_config_pb2.ModelConfig())
    response = service_pb2.ModelConfigResponse(config=model_config_protobuf)
    response_dict = json.loads(json_format.MessageToJson(response, preserving_proto_field_name=True))
    mock_get_model_config = mocker.patch.object(base_triton_client, base_triton_client.get_model_config.__name__)
    mock_get_model_config.return_value = response_dict


@wrapt.decorator
def patch_server_model_addsub_no_batch_ready(wrapped, _instance, _args, kwargs):
    mocker = kwargs["mocker"]
    patch_client__server_up_and_ready(mocker, SyncGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG, SyncGrpcInferenceServerClient)
    return wrapped(mocker)
