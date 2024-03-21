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

import asyncio
import gc
import logging
import threading
import unittest
from unittest.mock import ANY

import numpy as np
import pytest
from tritonclient.grpc.aio import InferenceServerClient as AsyncioGrpcInferenceServerClient
from tritonclient.http.aio import InferenceServerClient as AsyncioHttpInferenceServerClient

from pytriton.client import AsyncioDecoupledModelClient, AsyncioModelClient
from pytriton.client.asyncio_utils import asyncio_wait_for_model_ready
from pytriton.client.exceptions import (
    PyTritonClientInferenceServerError,
    PyTritonClientInvalidUrlError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)

from .client_common import (
    ADD_SUB_WITH_BATCHING_MODEL_CONFIG,
    ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG,
    ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG_DECOUPLED,
    EXPECTED_KWARGS_DEFAULT,
    GRPC_LOCALHOST_URL,
    HTTP_LOCALHOST_URL,
    HTTP_LOCALHOST_URL_NO_SCHEME,
    patch_client__server_up_and_ready,
    patch_grpc_client__model_up_and_ready,
    patch_http_client__model_up_and_ready,
)
from .utils import (
    extract_array_from_http_infer_input,
    verify_equalness_of_dicts_with_ndarray,
    wrap_to_http_infer_result,
)

_LOGGER = logging.getLogger(__name__)


async def test_utils_asyncio_wait_for_model_ready_http_client_not_ready_server(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient, ready_server=False)

    triton_client = AsyncioHttpInferenceServerClient(url=HTTP_LOCALHOST_URL_NO_SCHEME, verbose=False)
    try:
        with pytest.raises(PyTritonClientTimeoutError):
            await asyncio_wait_for_model_ready(
                asyncio_client=triton_client,
                model_name=ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                model_version=str(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_version),
                timeout_s=1,
            )
    finally:
        await triton_client.close()


async def test_utils_asyncio_wait_for_model_ready_http_client_not_live_server(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient, live_server=False)

    triton_client = AsyncioHttpInferenceServerClient(url=HTTP_LOCALHOST_URL_NO_SCHEME, verbose=False)
    try:
        with pytest.raises(PyTritonClientTimeoutError):
            await asyncio_wait_for_model_ready(
                asyncio_client=triton_client,
                model_name=ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                model_version=str(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_version),
                timeout_s=1,
            )
    finally:
        await triton_client.close()


async def test_utils_asyncio_wait_for_model_ready_http_client_model_not_ready(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient, ready=False
    )

    triton_client = AsyncioHttpInferenceServerClient(url=HTTP_LOCALHOST_URL_NO_SCHEME, verbose=False)
    try:
        with pytest.raises(PyTritonClientTimeoutError):
            await asyncio_wait_for_model_ready(
                asyncio_client=triton_client,
                model_name=ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
                model_version=str(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_version),
                timeout_s=1,
            )
    finally:
        await triton_client.close()


async def test_utils_asyncio_wait_for_model_ready_http_client_success(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient
    )

    triton_client = AsyncioHttpInferenceServerClient(url=HTTP_LOCALHOST_URL_NO_SCHEME, verbose=False)
    await asyncio_wait_for_model_ready(
        asyncio_client=triton_client,
        model_name=ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
        model_version=str(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_version),
        timeout_s=1,
    )
    await triton_client.close()


async def test_async_client_init_raises_error_when_invalid_url_provided(mocker):
    with pytest.raises(PyTritonClientInvalidUrlError):
        async with AsyncioModelClient(["localhost:8001"], "dummy") as _:  # pytype: disable=wrong-arg-types
            pass


async def test_async_http_client_init_raises_error_when_use_non_lazy_init_on_non_responding_server():
    with pytest.raises(PyTritonClientTimeoutError):
        async with AsyncioModelClient("dummy:43299", "dummy", lazy_init=False, init_timeout_s=1) as _:
            pass


async def test_async_http_client_init_obtain_expected_model_config_when_lazy_init_is_disabled(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient)

    spy_client_init = mocker.spy(AsyncioHttpInferenceServerClient, AsyncioHttpInferenceServerClient.__init__.__name__)
    client = AsyncioModelClient("http://localhost:8000", ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name, lazy_init=False)
    await client.__aenter__()
    # Exit sets some clients to none
    general_client = client._general_client
    infer_client = client._infer_client
    await client.__aexit__(None, None, None)
    assert spy_client_init.mock_calls == [
        unittest.mock.call(general_client, "localhost:8000", conn_timeout=60.0),
        unittest.mock.call(infer_client, "localhost:8000", conn_timeout=60.0),
    ]
    assert await client.model_config == ADD_SUB_WITH_BATCHING_MODEL_CONFIG


async def test_async_http_client_model_config_raises_error_when_requested_unavailable_model(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)

    mocker.patch.object(
        AsyncioHttpInferenceServerClient, AsyncioHttpInferenceServerClient.is_model_ready.__name__
    ).return_value = False

    with pytest.raises(PyTritonClientTimeoutError, match="Timeout while waiting for model"):
        async with AsyncioModelClient(HTTP_LOCALHOST_URL, "NonExistentModel", init_timeout_s=1) as client:
            _ = await client.model_config

    with pytest.raises(PyTritonClientTimeoutError, match="Timeout while waiting for model"):
        async with AsyncioModelClient(HTTP_LOCALHOST_URL, "OtherName", "2", init_timeout_s=1) as client:
            _ = await client.model_config


async def test_async_http_client_infer_raises_error_when_requested_unavailable_model(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    mocker.patch.object(
        AsyncioHttpInferenceServerClient, AsyncioHttpInferenceServerClient.is_model_ready.__name__
    ).return_value = False
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    with pytest.raises(PyTritonClientTimeoutError, match="Timeout while waiting for model"):
        async with AsyncioModelClient(HTTP_LOCALHOST_URL, "NonExistentModel", init_timeout_s=1) as client:
            _ = await client.infer_sample(a, b)

    with pytest.raises(PyTritonClientTimeoutError, match="Timeout while waiting for model"):
        async with AsyncioModelClient(HTTP_LOCALHOST_URL, "NonExistentModel", init_timeout_s=1) as client:
            _ = await client.infer_batch(a, b)

    with pytest.raises(PyTritonClientTimeoutError, match="Timeout while waiting for model"):
        async with AsyncioModelClient(HTTP_LOCALHOST_URL, "OtherName", "2", init_timeout_s=1) as client:
            _ = await client.infer_sample(a, b)

    with pytest.raises(PyTritonClientTimeoutError, match="Timeout while waiting for model"):
        async with AsyncioModelClient(HTTP_LOCALHOST_URL, "OtherName", "2", init_timeout_s=1) as client:
            _ = await client.infer_batch(a, b)


async def test_async_http_client_infer_sample_returns_expected_result_when_infer_on_model_with_batching(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    # server will return data with additional axis
    server_result = {name: data[np.newaxis, ...] for name, data in expected_result.items()}

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)
        result = await client.infer_sample(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_DEFAULT)
        expected_kwargs.update({
            # expect to send data with additional batch axis
            "inputs": {"a": a[np.newaxis, ...], "b": b[np.newaxis, ...]},
            "outputs": list(expected_result),
        })
        assert all(
            called_kwargs.get(arg_name) == arg_value
            for arg_name, arg_value in expected_kwargs.items()
            if arg_name not in ["inputs", "outputs"]  # inputs and outputs requires manual verification
        )
        assert not [key for key in called_kwargs if key not in list(expected_kwargs)]
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


async def test_async_http_client_infer_sample_fails_for_decoupled_model(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG_DECOUPLED, AsyncioHttpInferenceServerClient
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientInferenceServerError):
            await client.infer_sample(a, b)


async def test_async_http_client_infer_sample_fails_for_http_decoupled(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG_DECOUPLED, AsyncioHttpInferenceServerClient
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    async with AsyncioDecoupledModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError):
            # pytype: disable=attribute-error
            # Pytype fails to deduct that this is asyncio iterator
            async for _ in client.infer_sample(a, b):
                pass
            # pytype: enable=attribute-error


async def test_async_grpc_client_infer_sample_returns_expected_result_decoupled(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}

    model_config = ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG_DECOUPLED

    _LOGGER.debug("Creating client")
    client = AsyncioDecoupledModelClient(GRPC_LOCALHOST_URL, model_config.model_name)
    _LOGGER.debug("Creating client")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient)
    mock_infer = mocker.patch.object(client._infer_client, "stream_infer")

    async def result_generator():
        yield (wrap_to_http_infer_result(model_config, "0", expected_result), None)

    mock_infer.return_value = result_generator()
    _LOGGER.debug("Entering client")
    await client.__aenter__()
    _LOGGER.debug("Entered client")
    outside_result = None
    async for result in client.infer_sample(a, b):
        outside_result = result
    _LOGGER.debug("Exiting client")
    await client.__aexit__(None, None, None)
    _LOGGER.debug("Exited client")

    assert outside_result == expected_result


async def test_async_grpc_client_infer_sample_raises_exception_decoupled(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}

    model_config = ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG_DECOUPLED

    _LOGGER.debug("Creating client")
    client = AsyncioDecoupledModelClient(GRPC_LOCALHOST_URL, model_config.model_name)
    _LOGGER.debug("Creating client")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient)
    mock_infer = mocker.patch.object(client._infer_client, "stream_infer")

    async def result_generator():
        yield (wrap_to_http_infer_result(model_config, "0", expected_result), None)
        yield (None, PyTritonClientInferenceServerError("dummy"))

    mock_infer.return_value = result_generator()
    _LOGGER.debug("Entering client")
    await client.__aenter__()
    _LOGGER.debug("Entered client")
    outside_result = None
    with pytest.raises(PyTritonClientInferenceServerError):
        async for result in client.infer_sample(a, b):
            outside_result = result
    _LOGGER.debug("Exiting client")
    await client.__aexit__(None, None, None)
    _LOGGER.debug("Exited client")

    assert outside_result == expected_result


async def test_async_grpc_client_infer_sample_raises_exception_wrong_coupled_decoupled(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    model_config = ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioDecoupledModelClient(GRPC_LOCALHOST_URL, model_config.model_name)
    _LOGGER.debug("Creating client")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient)
    _LOGGER.debug("Entering client")
    await client.__aenter__()
    _LOGGER.debug("Entered client")
    with pytest.raises(PyTritonClientInferenceServerError):
        async for _result in client.infer_sample(a, b):
            pass
    _LOGGER.debug("Exiting client")
    await client.__aexit__(None, None, None)
    _LOGGER.debug("Exited client")


async def test_async_http_client_infer_sample_returns_expected_result_when_infer_on_model_with_batching_created_from_existing(
    mocker,
):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient)

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    # server will return data with additional axis
    server_result = {name: data[np.newaxis, ...] for name, data in expected_result.items()}

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)
        await client.infer_sample(a, b)

        async with AsyncioModelClient.from_existing_client(client) as client_from_existing:
            mock_infer_from_existing = mocker.patch.object(client_from_existing._infer_client, "infer")
            mock_infer_from_existing.return_value = wrap_to_http_infer_result(
                ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result
            )
            result = await client.infer_sample(a, b)
            verify_equalness_of_dicts_with_ndarray(expected_result, result)
        async with AsyncioModelClient(
            HTTP_LOCALHOST_URL,
            ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name,
            model_config=await client.model_config,
            ensure_model_is_ready=False,
        ) as client_from_existing:
            mock_infer_from_existing = mocker.patch.object(client_from_existing._infer_client, "infer")
            mock_infer_from_existing.return_value = wrap_to_http_infer_result(
                ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result
            )
            result = await client.infer_sample(a, b)
            verify_equalness_of_dicts_with_ndarray(expected_result, result)


async def test_async_http_client_infer_sample_returns_expected_result_when_positional_args_are_used(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, "0", server_result)
        result = await client.infer_sample(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_DEFAULT)
        expected_kwargs.update({
            "model_name": ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
            "inputs": {"a": a, "b": b},
            "outputs": list(expected_result),
        })
        assert all(
            called_kwargs.get(arg_name) == arg_value
            for arg_name, arg_value in expected_kwargs.items()
            if arg_name not in ["inputs", "outputs"]  # inputs and outputs requires manual verification
        )
        assert not [key for key in called_kwargs if key not in list(expected_kwargs)]
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


async def test_async_http_client_infer_batch_returns_expected_result_when_positional_args_are_used(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient)

    a = np.array([[1], [1]], dtype=np.float32)
    b = np.array([[1], [1]], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)
        result = await client.infer_batch(a, b)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_DEFAULT)
        expected_kwargs.update({
            "inputs": {"a": a, "b": b},
            "outputs": list(expected_result),
        })
        assert all(
            called_kwargs.get(arg_name) == arg_value
            for arg_name, arg_value in expected_kwargs.items()
            if arg_name not in ["inputs", "outputs"]  # inputs and outputs requires manual verification
        )
        assert not [key for key in called_kwargs if key not in list(expected_kwargs)]
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


async def test_async_http_client_infer_sample_returns_expected_result_when_named_args_are_used(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = {"add": a + b, "sub": a - b}

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = await client.infer_sample(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_DEFAULT)
        expected_kwargs.update({
            "model_name": ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name,
            "inputs": inputs_dict,
            "outputs": list(expected_result),
        })
        assert all(
            called_kwargs.get(arg_name) == arg_value
            for arg_name, arg_value in expected_kwargs.items()
            if arg_name not in ["inputs", "outputs"]  # inputs and outputs requires manual verification
        )
        assert not [key for key in called_kwargs if key not in list(expected_kwargs)]
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


async def test_async_http_client_infer_batch_returns_expected_result_when_named_args_are_used(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(mocker, ADD_SUB_WITH_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient)

    a = np.array([[1], [1]], dtype=np.float32)
    b = np.array([[1], [1]], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}
    server_result = expected_result

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        mock_infer = mocker.patch.object(client._infer_client, "infer")
        mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", server_result)

        inputs_dict = {"a": a, "b": b}
        result = await client.infer_batch(**inputs_dict)

        called_kwargs = mock_infer.call_args.kwargs
        expected_kwargs = dict(EXPECTED_KWARGS_DEFAULT)
        expected_kwargs.update({
            "inputs": inputs_dict,
            "outputs": list(expected_result),
        })
        assert all(
            called_kwargs.get(arg_name) == arg_value
            for arg_name, arg_value in expected_kwargs.items()
            if arg_name not in ["inputs", "outputs"]  # inputs and outputs requires manual verification
        )
        assert not [key for key in called_kwargs if key not in list(expected_kwargs)]
        assert [output.name() for output in called_kwargs.get("outputs")] == list(expected_kwargs["outputs"])
        inputs_called_arg = {i.name(): extract_array_from_http_infer_input(i) for i in called_kwargs.get("inputs")}
        verify_equalness_of_dicts_with_ndarray(inputs_called_arg, expected_kwargs["inputs"])

        verify_equalness_of_dicts_with_ndarray(expected_result, result)


async def test_async_http_client_infer_batch_raises_error_when_model_doesnt_support_batching(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientModelDoesntSupportBatchingError):
            await client.infer_batch(a, b)


async def test_async_http_client_infer_raises_error_when_mixed_args_convention_used(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient
    )

    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            await client.infer_sample(a, b=b)

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(
            PyTritonClientValueError,
            match="Use either positional either keyword method arguments convention",
        ):
            await client.infer_batch(a, b=b)


async def test_async_http_client_infer_raises_error_when_no_args_provided(mocker):
    patch_client__server_up_and_ready(mocker, AsyncioHttpInferenceServerClient)
    patch_http_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioHttpInferenceServerClient
    )

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError, match="Provide input data"):
            await client.infer_sample()

    async with AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name) as client:
        with pytest.raises(PyTritonClientValueError, match="Provide input data"):
            await client.infer_batch()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
async def test_asynciodel_of_inference_client_does_not_raise_error():
    def _del(client):
        del client._general_client
        del client._infer_client

    async def _create_client_and_delete():
        client = AsyncioModelClient(HTTP_LOCALHOST_URL, ADD_SUB_WITH_BATCHING_MODEL_CONFIG.model_name)
        await client.close()
        threading.Thread(target=_del, args=(client,)).start()

    await _create_client_and_delete()
    gc.collect()


async def test_async_grpc_client_infer_sample_returns_expected_result_when_infer_on_model_with_batching(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = {"add": a + b, "sub": a - b}

    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, model_config.model_name)
    _LOGGER.debug("Creating client")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(
        mocker, ADD_SUB_WITHOUT_BATCHING_MODEL_CONFIG, AsyncioGrpcInferenceServerClient
    )
    mock_infer = mocker.patch.object(client._infer_client, "infer")
    mock_infer.return_value = wrap_to_http_infer_result(ADD_SUB_WITH_BATCHING_MODEL_CONFIG, "0", expected_result)
    _LOGGER.debug("Entering client")
    await client.__aenter__()
    _LOGGER.debug("Entered client")
    result = await client.infer_sample(a, b)
    mock_infer.assert_called_with(
        model_name=model_config.model_name,
        model_version="",
        inputs=ANY,
        request_id=ANY,
        headers=None,
        parameters=None,
        outputs=ANY,
        client_timeout=60.0,
    )
    _LOGGER.debug("Exiting client")
    await client.__aexit__(None, None, None)
    _LOGGER.debug("Exited client")

    assert result == expected_result


async def test_async_grpc_client_non_lazy_aenter_failure_triton_non_ready(mocker):
    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, model_config.model_name, init_timeout_s=0.1, lazy_init=False)
    _LOGGER.debug("Before patching")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient, ready_server=False)
    _LOGGER.debug("Entering client")
    with pytest.raises(PyTritonClientTimeoutError):
        await asyncio.wait_for(client.__aenter__(), 0.2)
        _LOGGER.debug("Exiting client without error")
    _LOGGER.debug("Exited client with error")


async def test_async_grpc_client_non_lazy_aenter_failure_triton_non_live(mocker):
    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, model_config.model_name, init_timeout_s=0.1, lazy_init=False)
    _LOGGER.debug("Before patching")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient, live_server=False)
    _LOGGER.debug("Entering client")
    with pytest.raises(PyTritonClientTimeoutError):
        await asyncio.wait_for(client.__aenter__(), 0.2)
        _LOGGER.debug("Exiting client without error")
    _LOGGER.debug("Exited client with error")


async def test_async_grpc_client_non_lazy_aenter_failure_model_non_ready(mocker):
    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, model_config.model_name, init_timeout_s=0.1, lazy_init=False)
    _LOGGER.debug("Before patching")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient, ready=False)
    _LOGGER.debug("Entering client")
    with pytest.raises(PyTritonClientTimeoutError):
        await asyncio.wait_for(client.__aenter__(), 0.2)
        _LOGGER.debug("Exiting client without error")
    _LOGGER.debug("Exited client with error")


async def test_async_grpc_client_non_lazy_aenter_failure_model_state_unavailable(mocker):
    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, model_config.model_name, init_timeout_s=1, lazy_init=False)
    _LOGGER.debug("Before patching")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient, ready=False)
    _LOGGER.debug("Entering client")
    with pytest.raises(PyTritonClientTimeoutError):
        await asyncio.wait_for(client.__aenter__(), 2)
        _LOGGER.debug("Exiting client without error")
    _LOGGER.debug("Exited client with error")


async def test_async_grpc_client_non_lazy_aenter_failure_model_incorrect_name(mocker):
    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, "DUMMY", init_timeout_s=1, lazy_init=False)
    _LOGGER.debug("Before patching")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient)
    _LOGGER.debug("Entering client")
    with pytest.raises(PyTritonClientTimeoutError):
        await asyncio.wait_for(client.__aenter__(), 2)
        _LOGGER.debug("Exiting client without error")
    _LOGGER.debug("Exited client with error")


async def test_async_grpc_client_non_lazy_aenter_failure_model_incorrect_version(mocker):
    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(
        GRPC_LOCALHOST_URL, model_config.model_name, model_version="2", init_timeout_s=1, lazy_init=False
    )
    _LOGGER.debug("Before patching")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient)
    _LOGGER.debug("Entering client")
    with pytest.raises(PyTritonClientTimeoutError):
        await asyncio.wait_for(client.__aenter__(), 2)
        _LOGGER.debug("Exiting client without error")
    _LOGGER.debug("Exited client with error")


async def test_async_grpc_client_infer_sample_fails_on_model_with_batching(mocker):
    a = np.array([1], dtype=np.float32)
    b = np.array([1], dtype=np.float32)

    model_config = ADD_SUB_WITH_BATCHING_MODEL_CONFIG

    _LOGGER.debug("Creating client")
    client = AsyncioModelClient(GRPC_LOCALHOST_URL, model_config.model_name)
    _LOGGER.debug("Creating client")
    patch_client__server_up_and_ready(mocker, AsyncioGrpcInferenceServerClient)
    patch_grpc_client__model_up_and_ready(mocker, model_config, AsyncioGrpcInferenceServerClient)
    mock_infer = mocker.patch.object(client._infer_client, "infer")

    def _model_infer_mock(*args, **kwargs):
        raise PyTritonClientValueError("Dummy exception")

    mock_infer.side_effect = _model_infer_mock

    _LOGGER.debug("Entering client")
    await client.__aenter__()
    _LOGGER.debug("Entered client")

    with pytest.raises(PyTritonClientValueError):
        await client.infer_sample(a, b)

    _LOGGER.debug("Exiting client")
    await client.__aexit__(None, None, None)
    _LOGGER.debug("Exited client")


async def test_async_http_init_passes_timeout(mocker):
    async with AsyncioModelClient(
        "http://localhost:6669", "dummy", init_timeout_s=0.2, inference_timeout_s=0.1
    ) as client:
        with pytest.raises(PyTritonClientTimeoutError):
            await client.wait_for_model(timeout_s=0.2)


async def test_async_grpc_init_passes_timeout(mocker):
    async with AsyncioModelClient(
        "grpc://localhost:6669", "dummy", init_timeout_s=0.2, inference_timeout_s=0.1
    ) as client:
        with pytest.raises(PyTritonClientTimeoutError):
            await client.wait_for_model(timeout_s=0.2)
