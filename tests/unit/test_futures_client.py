# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import logging
from threading import Thread

import numpy as np
import pytest

from pytriton.client import FuturesModelClient, ModelClient
from pytriton.client.exceptions import (
    PyTritonClientClosedError,
    PyTritonClientInvalidUrlError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from pytriton.model_config import DeviceKind
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig

from .client_common import (
    ADD_SUB_WITH_BATCHING_MODEL_CONFIG,
    GRPC_LOCALHOST_URL,
    HTTP_LOCALHOST_URL,
    patch_server_model_addsub_no_batch_ready,
)
from .utils import (
    patch_grpc_client__model_up_and_ready,
    patch_grpc_client__server_up_and_ready,
    patch_http_client__model_up_and_ready,
    patch_http_client__server_up_and_ready,
)

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("test_sync_client")


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


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("test_sync_client")


def test_wait_for_model_raise_error_when_invalid_url_provided():
    with pytest.raises(PyTritonClientInvalidUrlError, match="Invalid url"):
        with FuturesModelClient(["localhost:8001"], "dummy") as client:  # pytype: disable=wrong-arg-types
            client.wait_for_model(timeout_s=0.1).result()


@patch_server_model_addsub_no_batch_ready
def test_wait_for_model_passes_timeout_to_client(mocker):
    spy_client_close = mocker.spy(ModelClient, ModelClient.close.__name__)
    mock_client_wait_for_model = mocker.patch.object(ModelClient, ModelClient.wait_for_model.__name__)
    mock_client_wait_for_model.return_value = True
    spy__thread_start = mocker.spy(Thread, Thread.start.__name__)
    with FuturesModelClient(
        GRPC_LOCALHOST_URL,
        ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
        str(ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_version),
        max_workers=1,
    ) as client:
        future = client.wait_for_model(15)
        result = future.result()
        assert result is True
    spy_client_close.assert_called_once()
    mock_client_wait_for_model.assert_called_with(15)
    spy__thread_start.assert_called_once()


@patch_server_model_addsub_no_batch_ready
def test_infer_raises_error_when_mixed_args_convention_used(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    init_t_timeout_s = 15.0

    with FuturesModelClient(
        GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, init_timeout_s=init_t_timeout_s
    ) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_sample(a, b=b).result()

    with FuturesModelClient(GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            client.infer_batch(a, b=b).result()


@patch_server_model_addsub_no_batch_ready
def test_infer_sample_returns_values_creates_client(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    c = np.array([3], dtype=np.float32)

    init_t_timeout_s = 15.0

    mock_client_wait_for_model = mocker.patch.object(ModelClient, ModelClient._wait_and_init_model_config.__name__)
    mock_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)

    mock_client_infer_sample.return_value = c
    with FuturesModelClient(
        GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, init_timeout_s=init_t_timeout_s
    ) as client:
        result = client.infer_sample(a=a, b=b).result()
    mock_client_wait_for_model.assert_called_once_with(init_t_timeout_s)
    mock_client_infer_sample.assert_called_once_with(parameters=None, headers=None, a=a, b=b)
    # Check the Python version and use different assertions for cancel_futures
    assert result == c


@patch_server_model_addsub_no_batch_ready
def test_infer_sample_returns_values_creates_client_close_wait(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    c = np.array([3], dtype=np.float32)

    mock_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)

    # Prevent exit from closing the client
    mocker.patch.object(FuturesModelClient, FuturesModelClient.__exit__.__name__)

    mock_client_infer_sample.return_value = c
    client = FuturesModelClient(GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name)
    result = client.infer_sample(a, b).result()
    client.close(wait=True)
    mock_client_infer_sample.assert_called_once_with(a, b, parameters=None, headers=None)
    assert result == c


@patch_server_model_addsub_no_batch_ready
def test_infer_batch_returns_values_creates_client(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    c = np.array([3], dtype=np.float32)

    init_t_timeout_s = 15.0

    mock_client_infer_batch = mocker.patch.object(ModelClient, ModelClient.infer_batch.__name__)
    mock_client_infer_batch.return_value = c
    with FuturesModelClient(
        GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, init_timeout_s=init_t_timeout_s
    ) as client:
        result = client.infer_batch(a=a, b=b).result()
        model_config = client.model_config().result()
    mock_client_infer_batch.assert_called_once_with(parameters=None, headers=None, a=a, b=b)
    assert model_config.model_name == ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name
    assert result == c


@patch_server_model_addsub_no_batch_ready
def test_infer_sample_list_passed_arguments_returns_arguments(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)
    patch_client_infer_sample.return_value = ret
    with FuturesModelClient(GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:

        return_value = client.infer_sample(a, b).result()
        assert return_value == ret
        patch_client_infer_sample.assert_called_once_with(a, b, parameters=None, headers=None)


@patch_server_model_addsub_no_batch_ready
def test_infer_sample_dict_passed_arguments_returns_arguments(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_sample = mocker.patch.object(ModelClient, ModelClient.infer_sample.__name__)
    patch_client_infer_sample.return_value = ret
    with FuturesModelClient(GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:

        return_value = client.infer_sample(a=a, b=b).result()
        assert return_value == ret
        patch_client_infer_sample.assert_called_once_with(a=a, b=b, parameters=None, headers=None)


@patch_server_model_addsub_no_batch_ready
def test_infer_batch_list_passed_arguments_returns_arguments(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_batch = mocker.patch.object(ModelClient, ModelClient.infer_batch.__name__)
    patch_client_infer_batch.return_value = ret
    with FuturesModelClient(GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        return_value = client.infer_batch(a, b).result()
        assert return_value == ret
        patch_client_infer_batch.assert_called_once_with(a, b, parameters=None, headers=None)


@patch_server_model_addsub_no_batch_ready
def test_infer_batch_dict_passed_arguments_returns_arguments(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([2], dtype=np.float32)
    ret = np.array([3], dtype=np.float32)

    patch_client_infer_batch = mocker.patch.object(ModelClient, ModelClient.infer_batch.__name__)
    patch_client_infer_batch.return_value = ret
    with FuturesModelClient(GRPC_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:

        return_value = client.infer_batch(a=a, b=b).result()
        assert return_value == ret
        patch_client_infer_batch.assert_called_once_with(parameters=None, headers=None, a=a, b=b)


@pytest.mark.timeout(0.3)
def test_init_http_passes_timeout(mocker):
    with FuturesModelClient("http://localhost:6669", "dummy", init_timeout_s=0.2, inference_timeout_s=0.1) as client:
        with pytest.raises(PyTritonClientTimeoutError):
            client.wait_for_model(timeout_s=0.2).result()


@pytest.mark.timeout(5)
def test_init_grpc_passes_timeout_5(mocker):
    with FuturesModelClient("grpc://localhost:6669", "dummy", init_timeout_s=0.2, inference_timeout_s=0.1) as client:
        with pytest.raises(PyTritonClientTimeoutError):
            client.wait_for_model(timeout_s=0.2).result()


@pytest.mark.timeout(5)
def test_init_grpc_spaws_5_threads(mocker):
    spy_thread_start = mocker.spy(Thread, Thread.start.__name__)

    with FuturesModelClient("grpc://localhost:6669", "dummy", init_timeout_s=1, inference_timeout_s=0.1) as client:
        timeout_s = 0.2
        with pytest.raises(PyTritonClientTimeoutError):
            # The list function is used to force the evaluation of the list comprehension before iterating over the futures and
            # calling their result method. This is done to ensure that all the calls occur before the iteration starts,
            # and to verify that five threads are created.
            futures = list([client.wait_for_model(timeout_s=timeout_s) for _ in range(5)])  # noqa: C411
            for future in futures:
                future.result()
        assert spy_thread_start.call_count == 5


def test_http_client_raises_error_when_used_after_close(mocker):
    patch_http_client__server_up_and_ready(mocker)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG)

    with ModelClient(HTTP_LOCALHOST_URL, "dummy") as client:
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

    with FuturesModelClient(GRPC_LOCALHOST_URL, "dummy") as client:
        pass

    with pytest.raises(PyTritonClientClosedError):
        client.wait_for_model(timeout_s=0.2).result()

    a = np.array([1], dtype=np.float32)
    with pytest.raises(PyTritonClientClosedError):
        client.infer_sample(a=a).result()

    with pytest.raises(PyTritonClientClosedError):
        client.infer_batch(a=[a]).result()
