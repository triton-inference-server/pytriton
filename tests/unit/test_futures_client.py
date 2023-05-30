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
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from tritonclient.grpc import InferenceServerClient as GrpcInferenceServerClient

from pytriton.client import FuturesModelClient, ModelClient
from pytriton.client.exceptions import PyTritonClientUrlParseError, PyTritonClientValueError
from pytriton.model_config import DeviceKind
from pytriton.model_config.generator import ModelConfigGenerator
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("test_sync_client")

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
    backend_parameters={"shared-memory-socket": "dummy/path"},
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
    backend_parameters={"shared-memory-socket": "dummy/path"},
)

_GRPC_LOCALHOST_URL = "grpc://localhost:8001"


def _patch_grpc_client__server_up_and_ready(mocker):
    mocker.patch.object(
        GrpcInferenceServerClient, GrpcInferenceServerClient.is_server_ready.__name__
    ).return_value = True
    mocker.patch.object(
        GrpcInferenceServerClient, GrpcInferenceServerClient.is_server_live.__name__
    ).return_value = True


def _patch_grpc_client__model_up_and_ready(mocker, model_config: TritonModelConfig):
    from google.protobuf import json_format  # pytype: disable=pyi-error
    from tritonclient.grpc import model_config_pb2, service_pb2  # pytype: disable=pyi-error

    mock_get_repo_index = mocker.patch.object(
        GrpcInferenceServerClient, GrpcInferenceServerClient.get_model_repository_index.__name__
    )
    mock_get_repo_index.return_value = service_pb2.RepositoryIndexResponse(
        models=[
            service_pb2.RepositoryIndexResponse.ModelIndex(
                name=model_config.model_name, version="1", state="READY", reason=""
            ),
        ]
    )

    mocker.patch.object(
        GrpcInferenceServerClient, GrpcInferenceServerClient.is_model_ready.__name__
    ).return_value = True

    model_config_dict = ModelConfigGenerator(model_config).get_config()
    model_config_protobuf = json_format.ParseDict(model_config_dict, model_config_pb2.ModelConfig())
    response = service_pb2.ModelConfigResponse(config=model_config_protobuf)
    response_dict = json.loads(json_format.MessageToJson(response, preserving_proto_field_name=True))
    mock_get_model_config = mocker.patch.object(
        GrpcInferenceServerClient, GrpcInferenceServerClient.get_model_config.__name__
    )
    mock_get_model_config.return_value = response_dict


def test_wait_for_model_raise_error_when_invalid_url_provided():
    with pytest.raises(PyTritonClientUrlParseError, match="Could not parse url"):
        client = FuturesModelClient(["localhost:8001"], "dummy")  # pytype: disable=wrong-arg-types
        client.wait_for_model().result()


def test_wait_for_model_passes_timeout_to_client(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    spy_client_close = mocker.spy(ModelClient, ModelClient.close.__name__)
    mock_client_wait_for_model = mocker.patch.object(ModelClient, ModelClient.wait_for_model.__name__)
    mock_client_wait_for_model.return_value = True
    spy_thread_pool_executor_shutdown = mocker.spy(ThreadPoolExecutor, ThreadPoolExecutor.shutdown.__name__)
    spy_thread_pool_executor_submit = mocker.spy(ThreadPoolExecutor, ThreadPoolExecutor.submit.__name__)
    with FuturesModelClient(
        _GRPC_LOCALHOST_URL,
        ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
        str(ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_version),
        max_workers=1,
    ) as client:
        future = client.wait_for_model(15)
        result = future.result()
        assert result is True
    spy_client_close.assert_called_once()
    mock_client_wait_for_model.assert_called_with(15)
    spy_thread_pool_executor_shutdown.assert_called_once()
    spy_thread_pool_executor_submit.assert_called_once()


def test_init_passes_max_workers_to_thread_pool_executor(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    # Disable sync client exit
    mock_client = mocker.patch.object(ModelClient, ModelClient.__exit__.__name__)
    mock_threads_init = mocker.patch.object(ThreadPoolExecutor, "__init__", autospec=True)
    mock_threads_init.return_value = None
    # Disable thread pool executor shutdown
    mocker.patch.object(FuturesModelClient, FuturesModelClient.__exit__.__name__)
    client = FuturesModelClient(
        _GRPC_LOCALHOST_URL,
        ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
        str(ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_version),
        max_workers=2,
    )

    mock_threads_init.assert_called_once_with(client._thread_pool_executor, max_workers=2)
    mock_client.assert_not_called()


def test_infer_raises_error_when_mixed_args_convention_used(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    init_t_timeout_s = 15.0

    with FuturesModelClient(
        _GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, init_timeout_s=init_t_timeout_s
    ) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):

            client.infer_sample(a, b=b).result()

    with FuturesModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_batch(a, b=b).result()


def test_infer_sample_returns_values_creates_client(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    c = np.array([3], dtype=np.float32)

    init_t_timeout_s = 15.0

    mock_client_wait_for_model = mocker.patch.object(ModelClient, ModelClient._wait_and_init_model_config.__name__)
    mock_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)
    mock_thread_pool_executor_shutdown = mocker.patch.object(ThreadPoolExecutor, ThreadPoolExecutor.shutdown.__name__)

    mock_client_infer_sample.return_value = c
    with FuturesModelClient(
        _GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, init_timeout_s=init_t_timeout_s
    ) as client:
        result = client.infer_sample(a=a, b=b).result()
    mock_client_wait_for_model.assert_called_once_with(init_t_timeout_s)
    mock_client_infer_sample.assert_called_once_with(parameters=None, headers=None, a=a, b=b)
    # Check the Python version and use different assertions for cancel_futures
    mock_thread_pool_executor_shutdown.assert_called_once_with(wait=True)
    assert result == c


def test_infer_sample_returns_values_creates_client_close_wait(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    c = np.array([3], dtype=np.float32)

    mock_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)
    mock_thread_pool_executor_shutdown = mocker.patch.object(ThreadPoolExecutor, ThreadPoolExecutor.shutdown.__name__)

    # Prevent exit from closing the client
    mocker.patch.object(FuturesModelClient, FuturesModelClient.__exit__.__name__)

    mock_client_infer_sample.return_value = c
    client = FuturesModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name)
    result = client.infer_sample(a, b).result()
    client.close(wait=True, cancel_futures=True)
    mock_client_infer_sample.assert_called_once_with(a, b, parameters=None, headers=None)
    # Check the Python version and use different assertions for cancel_futures
    if sys.version_info >= (3, 9):
        mock_thread_pool_executor_shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    else:
        mock_thread_pool_executor_shutdown.assert_called_once_with(wait=True)
    assert result == c


def test_infer_batch_returns_values_creates_client(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    c = np.array([3], dtype=np.float32)

    init_t_timeout_s = 15.0

    mock_client_infer_batch = mocker.patch.object(ModelClient, ModelClient.infer_batch.__name__)
    mock_client_infer_batch.return_value = c
    with FuturesModelClient(
        _GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, init_timeout_s=init_t_timeout_s
    ) as client:
        result = client.infer_batch(a=a, b=b).result()
        model_config = client.model_config().result()
    mock_client_infer_batch.assert_called_once_with(parameters=None, headers=None, a=a, b=b)
    assert model_config.model_name == ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name
    assert result == c


def test_infer_sample_list_passed_arguments_returns_arguments(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)
    patch_client_infer_sample.return_value = ret
    with FuturesModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:

        return_value = client.infer_sample(a, b).result()
        assert return_value == ret
        patch_client_infer_sample.assert_called_once_with(a, b, parameters=None, headers=None)


def test_infer_sample_dict_passed_arguments_returns_arguments(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)
    patch_client_infer_sample.return_value = ret
    with FuturesModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:

        return_value = client.infer_sample(a=a, b=b).result()
        assert return_value == ret
        patch_client_infer_sample.assert_called_once_with(a=a, b=b, parameters=None, headers=None)


def test_infer_batch_list_passed_arguments_returns_arguments(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_batch = mocker.patch.object(ModelClient, ModelClient.infer_batch.__name__)
    patch_client_infer_batch.return_value = ret
    with FuturesModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        return_value = client.infer_batch(a, b).result()
        assert return_value == ret
        patch_client_infer_batch.assert_called_once_with(a, b, parameters=None, headers=None)


def test_infer_batch_dict_passed_arguments_returns_arguments(mocker):
    _patch_grpc_client__server_up_and_ready(mocker)
    _patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_batch = mocker.patch.object(ModelClient, ModelClient.infer_batch.__name__)
    patch_client_infer_batch.return_value = ret
    with FuturesModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:

        return_value = client.infer_batch(a=a, b=b).result()
        assert return_value == ret
        patch_client_infer_batch.assert_called_once_with(parameters=None, headers=None, a=a, b=b)
