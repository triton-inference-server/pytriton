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
import gc
import logging
import threading
import time

import numpy as np
import pytest
import tritonclient.grpc
import tritonclient.http

from pytriton.client import ModelClient
from pytriton.client.exceptions import (
    PyTritonClientClosedError,
    PyTritonClientInvalidUrlError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientModelUnavailableError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from pytriton.client.utils import _DEFAULT_NETWORK_TIMEOUT_S
from pytriton.model_config import DeviceKind
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig

from .utils import (
    extract_array_from_grpc_infer_input,
    extract_array_from_http_infer_input,
    patch_grpc_client__model_up_and_ready,
    patch_grpc_client__server_up_and_ready,
    patch_http_client__model_up_and_ready,
    patch_http_client__server_up_and_ready,
    verify_equalness_of_dicts_with_ndarray,
    wrap_to_grpc_infer_result,
    wrap_to_http_infer_result,
)

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
_HTTP_LOCALHOST_URL = "http://localhost:8000"


EXPECTED_KWARGS_HTTP_DEFAULT = {
    "model_name": ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
    "model_version": "",
    "request_id": "0",
    "parameters": None,
    "headers": None,
}  # Network timeout is passed to __init__ for client and applied to all network requests for HTTP sync client

EXPECTED_KWARGS_GRPC_DEFAULT = {
    **dict(EXPECTED_KWARGS_HTTP_DEFAULT.items()),
    "client_timeout": 60.0,  # Network timeout shall be passed always for GRPC sync client
}


def test_sync_client_init_raises_error_when_invalid_url_provided():
    with pytest.raises(PyTritonClientInvalidUrlError, match="Invalid url"):
        ModelClient(["localhost:8001"], "dummy")  # pytype: disable=wrong-arg-types


def test_sync_grpc_client_init_raises_error_when_use_non_lazy_init_on_non_responding_server():
    with pytest.raises(PyTritonClientTimeoutError, match="Waiting for (.*) to be ready timed out."):
        ModelClient("dummy:43299", "dummy", lazy_init=False, init_timeout_s=1)


def test_sync_grpc_client_init_raises_error_when_requested_unavailable_model_and_non_lazy_init_called(mocker):
    from tritonclient.grpc import service_pb2

    patch_grpc_client__server_up_and_ready(mocker)
    mock_get_repo_index = mocker.patch.object(tritonclient.grpc.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = service_pb2.RepositoryIndexResponse(
        models=[
            service_pb2.RepositoryIndexResponse.ModelIndex(name="OtherName", version="1", state="READY", reason=""),
        ]
    )

    with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
        ModelClient(_GRPC_LOCALHOST_URL, "NotExistentModel", lazy_init=False, init_timeout_s=10)

    with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
        ModelClient(_GRPC_LOCALHOST_URL, "OtherName", "2", lazy_init=False, init_timeout_s=10)


def test_sync_grpc_client_init_obtain_expected_model_config_when_lazy_init_is_disabled(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    spy_client_init = mocker.spy(tritonclient.grpc.InferenceServerClient, "__init__")
    spy_get_model_config = mocker.spy(tritonclient.grpc.InferenceServerClient, "get_model_config")
    client = ModelClient("grpc://localhost:8001", ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, lazy_init=False)

    assert [(call.args, call.kwargs) for call in spy_client_init.mock_calls] == [
        (
            (
                client._general_client,
                "localhost:8001",
            ),
            {},
        ),
        (
            (
                client._infer_client,
                "localhost:8001",
            ),
            {},
        ),
    ]

    spy_get_model_config.assert_called_once_with(
        ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
        "",
        as_json=True,
        # FIXME: GRPC client get_model_config doesn't support client_timeout parameter
        # client_timeout=60.0,
    )
    assert client.model_config == ADD_SUB_WITH_BATCHING_MODEL_CONFIG


def test_sync_grpc_client_model_config_raises_error_when_requested_unavailable_model(mocker):
    from tritonclient.grpc import service_pb2

    patch_grpc_client__server_up_and_ready(mocker)
    mock_get_repo_index = mocker.patch.object(tritonclient.grpc.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = service_pb2.RepositoryIndexResponse(
        models=[
            service_pb2.RepositoryIndexResponse.ModelIndex(name="OtherName", version="1", state="READY", reason=""),
        ]
    )
    with ModelClient(_GRPC_LOCALHOST_URL, "NonExistentModel") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.model_config

    with ModelClient(_GRPC_LOCALHOST_URL, "OtherName", "2") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.model_config


def test_sync_grpc_client_infer_raises_error_when_requested_unavailable_model(mocker):
    from tritonclient.grpc import service_pb2

    patch_grpc_client__server_up_and_ready(mocker)
    mock_get_repo_index = mocker.patch.object(tritonclient.grpc.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = service_pb2.RepositoryIndexResponse(
        models=[
            service_pb2.RepositoryIndexResponse.ModelIndex(name="OtherName", version="1", state="READY", reason=""),
        ]
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with ModelClient(_GRPC_LOCALHOST_URL, "NonExistentModel") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_sample(a, b)

    with ModelClient(_GRPC_LOCALHOST_URL, "NonExistentModel") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_batch(a, b)

    with ModelClient(_GRPC_LOCALHOST_URL, "OtherName", "2") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_sample(a, b)

    with ModelClient(_GRPC_LOCALHOST_URL, "OtherName", "2") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_batch(a, b)


def test_sync_grpc_client_infer_sample_returns_expected_result_when_positional_args_are_used(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_grpc_infer_result(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, "0", server_result)
        result = client.infer_sample(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_GRPC_DEFAULT)
        expected_kwargs.update(
            {
                "model_name": ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                "model_version": "",
                "request_id": "0",
                "inputs": {"a": a, "b": b},
                "outputs": list(expected_result),
                "parameters": None,
                "headers": None,
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_grpc_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])
        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_grpc_client_infer_sample_returns_expected_result_when_infer_on_model_with_batching(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    # server will return data with additional axis
    server_result = {name: data[np.newaxis, ...] for name, data in expected_result.items()}

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_grpc_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = client.infer_sample(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_GRPC_DEFAULT)
        expected_kwargs.update(
            {
                "model_name": ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
                # expect to send data with additional batch axis
                "inputs": {name: data[np.newaxis, ...] for name, data in inputs_dict.items()},
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_grpc_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_grpc_client_infer_sample_returns_expected_result_when_named_args_are_used(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = {"add": a + b, "sub": a - b}

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_grpc_infer_result(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = client.infer_sample(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_GRPC_DEFAULT)
        expected_kwargs.update(
            {
                "model_name": ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                "inputs": inputs_dict,
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_grpc_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_grpc_client_infer_batch_returns_expected_result_when_positional_args_are_used(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    a = np.array([[1], [1]], dtype=np.float32)
    b = np.array([[1], [1]], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_grpc_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)
        result = client.infer_batch(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_GRPC_DEFAULT)
        expected_kwargs.update(
            {
                "inputs": {"a": a, "b": b},
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_grpc_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_grpc_client_infer_batch_returns_expected_result_when_named_args_are_used(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    a = np.array([[1], [1]], dtype=np.float32)
    b = np.array([[1], [1]], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_grpc_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = client.infer_batch(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_GRPC_DEFAULT)
        expected_kwargs.update(
            {
                "inputs": inputs_dict,
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_grpc_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_grpc_client_infer_batch_raises_error_when_model_doesnt_support_batching(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientModelDoesntSupportBatchingError):
            client.infer_batch(a=a, b=b)


def test_sync_grpc_client_infer_raises_error_when_mixed_args_convention_used(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_sample(a, b=b)

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_batch(a, b=b)


def test_sync_grpc_client_infer_raises_error_when_no_args_provided(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError, match="Provide input data"):
            client.infer_sample()

    with ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError, match="Provide input data"):
            client.infer_batch()


def test_sync_http_client_init_obtain_expected_model_config_when_lazy_init_is_disabled(mocker):
    from pytriton.client.client import DEFAULT_INFERENCE_TIMEOUT_S

    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    spy_client_init = mocker.spy(tritonclient.http.InferenceServerClient, "__init__")
    client = ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, lazy_init=False)

    assert [(call.args, call.kwargs) for call in spy_client_init.mock_calls] == [
        (
            (client._general_client, "localhost:8000"),
            {"connection_timeout": _DEFAULT_NETWORK_TIMEOUT_S, "network_timeout": _DEFAULT_NETWORK_TIMEOUT_S},
        ),
        (
            (client._infer_client, "localhost:8000"),
            {"connection_timeout": DEFAULT_INFERENCE_TIMEOUT_S, "network_timeout": DEFAULT_INFERENCE_TIMEOUT_S},
        ),
    ]
    assert client.model_config == ADD_SUB_WITH_BATCHING_MODEL_CONFIG


def test_sync_http_client_init_raises_error_when_use_non_lazy_init():
    with pytest.raises(PyTritonClientTimeoutError, match="Waiting for (.*) to be ready timed out."):
        ModelClient("http://dummy:43299", "dummy", lazy_init=False, init_timeout_s=1)


def test_sync_http_client_init_raises_error_when_requested_unavailable_model_and_non_lazy_init_called(mocker):
    patch_http_client__server_up_and_ready(mocker)
    mock_get_repo_index = mocker.patch.object(tritonclient.http.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = [{"name": "OtherName", "version": "1", "state": "READY", "reason": ""}]

    with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
        ModelClient(_HTTP_LOCALHOST_URL, "NotExistentModel", lazy_init=False, init_timeout_s=10)

    with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
        ModelClient(_HTTP_LOCALHOST_URL, "OtherName", "2", lazy_init=False, init_timeout_s=10)


def test_sync_http_client_model_config_raises_error_when_requested_unavailable_model(mocker):
    patch_http_client__server_up_and_ready(mocker)
    mock_get_repo_index = mocker.patch.object(tritonclient.http.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = [{"name": "OtherName", "version": "1", "state": "READY", "reason": ""}]

    with ModelClient(_HTTP_LOCALHOST_URL, "NonExistentModel") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.model_config

    with ModelClient(_HTTP_LOCALHOST_URL, "OtherName", "2") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.model_config


def test_sync_http_client_infer_raises_error_when_requested_unavailable_model(mocker):
    patch_http_client__server_up_and_ready(mocker)
    mock_get_repo_index = mocker.patch.object(tritonclient.http.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = [{"name": "OtherName", "version": "1", "state": "READY", "reason": ""}]

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with ModelClient(_HTTP_LOCALHOST_URL, "NonExistentModel") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_sample(a, b)

    with ModelClient(_HTTP_LOCALHOST_URL, "NonExistentModel") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_batch(a, b)

    with ModelClient(_HTTP_LOCALHOST_URL, "OtherName", "2") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_sample(a, b)

    with ModelClient(_HTTP_LOCALHOST_URL, "OtherName", "2") as client:
        with pytest.raises(PyTritonClientModelUnavailableError, match="Model (.*) is unavailable."):
            _ = client.infer_batch(a, b)


def test_sync_http_client_infer_sample_returns_expected_result_when_infer_on_model_with_batching(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    # server will return data with additional axis
    server_result = {name: data[np.newaxis, ...] for name, data in expected_result.items()}

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)
        result = client.infer_sample(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_HTTP_DEFAULT)
        expected_kwargs.update(
            {
                # expect to send data with additional batch axis
                "inputs": {"a": a[np.newaxis, ...], "b": b[np.newaxis, ...]},
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_http_client_infer_sample_returns_expected_result_when_positional_args_are_used(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, "0", server_result)
        result = client.infer_sample(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_HTTP_DEFAULT)
        expected_kwargs.update(
            {
                "model_name": ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                "inputs": {"a": a, "b": b},
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_http_client_infer_sample_returns_expected_result_when_named_args_are_used(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = {"add": a + b, "sub": a - b}

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = client.infer_sample(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_HTTP_DEFAULT)
        expected_kwargs.update(
            {
                "model_name": ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                "inputs": inputs_dict,
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_http_client_infer_batch_returns_expected_result_when_positional_args_are_used(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    a = np.array([[1], [1]], dtype=np.float32)
    b = np.array([[1], [1]], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)
        result = client.infer_batch(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_HTTP_DEFAULT)
        expected_kwargs.update(
            {
                "inputs": {"a": a, "b": b},
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_http_client_infer_batch_returns_expected_result_when_named_args_are_used(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    a = np.array([[1], [1]], dtype=np.float32)
    b = np.array([[1], [1]], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = client.infer_batch(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_HTTP_DEFAULT)
        expected_kwargs.update(
            {
                "inputs": inputs_dict,
                "outputs": list(expected_result),
            }
        )
        for arg_name, arg_value in expected_kwargs.items():
            if arg_name not in ["inputs", "outputs"]:  # inputs and outputs requires manual verification
                assert called_kwargs.get(arg_name) == arg_value
        for key in called_kwargs:
            assert key in expected_kwargs
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


def test_sync_http_client_infer_batch_raises_error_when_model_doesnt_support_batching(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientModelDoesntSupportBatchingError):
            client.infer_batch(a, b)


def test_sync_http_client_infer_raises_error_when_mixed_args_convention_used(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_sample(a, b=b)

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_batch(a, b=b)


def test_sync_http_client_infer_raises_error_when_no_args_provided(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG)

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError, match="Provide input data"):
            client.infer_sample()

    with ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError, match="Provide input data"):
            client.infer_batch()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_del_of_http_client_does_not_raise_error():
    def _del(client):
        del client._general_client
        del client._infer_client

    def _create_client_and_delete():
        client = ModelClient(_HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name)
        client.close()
        threading.Thread(target=_del, args=(client,)).start()

    _create_client_and_delete()
    time.sleep(0.1)
    gc.collect()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_del_of_grpc_client_does_not_raise_error():
    def _del(client):
        del client._general_client
        del client._infer_client

    def _create_client_and_delete():
        client = ModelClient(_GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name)
        client.close()
        threading.Thread(target=_del, args=(client,)).start()

    _create_client_and_delete()
    time.sleep(0.1)
    gc.collect()


@pytest.mark.timeout(1.0)
def test_init_http_passes_timeout():
    with ModelClient("http://localhost:6669", "dummy", init_timeout_s=0.2, inference_timeout_s=0.1) as client:
        with pytest.raises(PyTritonClientTimeoutError):
            client.wait_for_model(timeout_s=0.2)


@pytest.mark.timeout(0.3)
def test_init_grpc_passes_timeout_03():
    with ModelClient("grpc://localhost:6669", "dummy", init_timeout_s=0.2, inference_timeout_s=0.1) as client:
        with pytest.raises(PyTritonClientTimeoutError):
            client.wait_for_model(timeout_s=0.2)


def test_http_client_raises_error_when_used_after_close(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    with ModelClient(_HTTP_LOCALHOST_URL, "dummy") as client:
        pass

    with pytest.raises(PyTritonClientClosedError):
        client.wait_for_model(timeout_s=0.2)

    a = np.array([1], dtype=np.float32)
    with pytest.raises(PyTritonClientClosedError):
        client.infer_sample(a=a)

    with pytest.raises(PyTritonClientClosedError):
        client.infer_batch(a=[a])


def test_grpc_client_raises_error_when_used_after_close(mocker):
    patch_grpc_client__server_up_and_ready(mocker)
    patch_grpc_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    with ModelClient(_GRPC_LOCALHOST_URL, "dummy") as client:
        pass

    with pytest.raises(PyTritonClientClosedError):
        client.wait_for_model(timeout_s=0.2)

    a = np.array([1], dtype=np.float32)
    with pytest.raises(PyTritonClientClosedError):
        client.infer_sample(a=a)

    with pytest.raises(PyTritonClientClosedError):
        client.infer_batch(a=[a])
