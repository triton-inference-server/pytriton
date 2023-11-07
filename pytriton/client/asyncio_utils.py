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
"""Utility module supporting model clients."""
import asyncio
import logging
import time
from typing import Optional, Union

import aiohttp
import grpc
import tritonclient.grpc
import tritonclient.http

from pytriton.client.exceptions import PyTritonClientModelUnavailableError, PyTritonClientTimeoutError
from pytriton.client.utils import LATEST_MODEL_VERSION, ModelState, parse_grpc_response, parse_http_response
from pytriton.model_config.parser import ModelConfigParser

aio_clients = Union[tritonclient.grpc.aio.InferenceServerClient, tritonclient.http.aio.InferenceServerClient]

_LOGGER = logging.getLogger(__name__)

_DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S = 60.0  # 60 seconds
_DEFAULT_ASYNC_SLEEP_FACTOR_S = 0.1  # 10% of timeout


async def asyncio_get_model_state(
    client: aio_clients,
    model_name: str,
    model_version: Optional[str] = None,
) -> ModelState:
    """Obtains state of the model deployed in Triton Inference Server.

    Typical use:

    >>> import tritonclient.http.aio
    ... client = tritonclient.http.aio.InferenceServerClient("localhost:8000")
    ... model_state = await get_model_state(client, "MyModel", "1")

    Args:
        client: Triton Inference Server client to use for communication
        model_name: name of the model which state we're requesting.
        model_version:
            version of the model which state we're requesting.
            If model_version is None state of latest model is returned.
            The latest versions of the model are the numerically greatest version numbers.

    Returns:
        Model state. ModelState.UNAVAILABLE is returned in case if model with given name and version is not found.

    """
    _LOGGER.debug(f"Obtaining model {model_name} state")
    repository_index = await client.get_model_repository_index()
    _LOGGER.debug("Model repository index obtained")
    if isinstance(repository_index, list):
        models_states = parse_http_response(models=repository_index)
    else:
        models_states = parse_grpc_response(models=repository_index.models)

    if model_version is None:
        requested_model_states = {
            version: state for (name, version), state in models_states.items() if name == model_name
        }
        if not requested_model_states:
            return ModelState.UNAVAILABLE
        else:
            requested_model_states = sorted(requested_model_states.items(), key=lambda item: int(item[0]))
            latest_version, latest_version_state = requested_model_states[-1]
            _LOGGER.debug(f"Model {model_name} latest version: {latest_version} state: {latest_version_state}")
            return latest_version_state
    else:
        key = (model_name, model_version)
        if key not in models_states:
            return ModelState.UNAVAILABLE
        else:
            model_state = models_states[key]
            _LOGGER.debug(f"Model {model_name} version {model_version} state: {model_state}")
            return model_state


async def asyncio_get_model_config(
    client: aio_clients,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
):
    """Obtain configuration of model deployed on the Triton Inference Server.

    Function waits for server readiness.

    Args:
        client: Triton Inference Server client to use for communication
        model_name: name of the model which configuration we're requesting.
        model_version:
            version of the model which configuration we're requesting.
            If model_version is None configuration of the latest model is returned.
            The latest versions of the model are the numerically greatest version numbers.
        timeout_s: timeout to finish model configuration obtain.

    Returns:
        Configuration of requested model.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
        PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    should_finish_before = time.time() + timeout_s
    _LOGGER.debug(f"Obtaining model {model_name} config (timeout={timeout_s:0.2f})")
    try:
        _LOGGER.debug(f"Waiting for model {model_name} to be ready")
        await asyncio.wait_for(
            asyncio_wait_for_model_ready(
                client, model_name=model_name, model_version=model_version, timeout_s=timeout_s
            ),
            timeout_s,
        )

        model_version = model_version or ""

        timeout_s = max(0, should_finish_before - time.time())
        if isinstance(client, tritonclient.grpc.aio.InferenceServerClient):
            _LOGGER.debug(f"Obtaining model {model_name} config as_json=True")
            response = await asyncio.wait_for(
                client.get_model_config(model_name, model_version, as_json=True), timeout_s
            )
            model_config = response["config"]
        else:
            _LOGGER.debug(f"Obtaining model {model_name} config")
            model_config = await asyncio.wait_for(client.get_model_config(model_name, model_version), timeout_s)
        _LOGGER.debug("Model config obtained")
        model_config = ModelConfigParser.from_dict(model_config)
        _LOGGER.debug(f"Model config: {model_config}")
        return model_config
    except asyncio.TimeoutError as e:
        message = f"Timeout while waiting for model {model_name} config (timeout={timeout_s:0.2f})"
        _LOGGER.error(message)
        raise PyTritonClientTimeoutError(message) from e


async def asyncio_wait_for_server_ready(
    asyncio_client: aio_clients,
    sleep_time_s: float,
):
    """Wait for Triton Inference Server readiness.

    There are two functions, which check server status:
    * asyncio_client.is_server_ready()
    * asyncio_client.is_server_live()
    Both must return true to consider server accessible to read model status.

    Function contains while loop with sleep to check server status periodically.

    Args:
        asyncio_client: Triton Inference Server client to use for communication
        sleep_time_s: time to sleep between server status checks

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
    """
    _LOGGER.debug("Waiting for server to be ready")
    try:
        while True:
            try:
                _LOGGER.debug("Waiting for server to be ready")
                server_ready = await asyncio_client.is_server_ready()
                _LOGGER.debug("Waiting for server to be live")
                server_live = await asyncio_client.is_server_live()
            except tritonclient.utils.InferenceServerException:
                # Raised by tritonclient/grpc/__init__.py:75
                server_live = False
                server_ready = False
            except aiohttp.client_exceptions.ClientConnectorError:
                # This exception is raised by aiohttp/connector.py:901 in _create_direct_connection
                # and it is not translated to any other error by tritonclient/http/aio/__init__.py:132 in _get method.
                #    res = await self._stub.get(url=req_url,
                # and tritonclient/http/aio/__init__.py:242 in is_server_ready method.
                #    response = await self._get(request_uri=request_uri,
                server_live = False
                server_ready = False
            except RuntimeError:
                # This exception is raised by aiohttp/client.py:400 in _request
                # and it is not translated to any other error by tritonclient/grpc/aio/__init__.py:151: in is_server_ready method.
                #    response = await self._client_stub.ServerReady(request=request,
                server_live = False
                server_ready = False
            except grpc._cython.cygrpc.UsageError:
                # This exception is raised by grpcio/grpc/_cython/_cygrpc/aio/channel.pyx.pxi:124
                # and it is not translated to any other error by tritonclient/grpc/aio/__init__.py", line 151, in is_server_ready
                #   response = await self._client_stub.ServerReady(request=request,
                server_live = False
                server_ready = False
            if server_ready and server_live:
                break
            _LOGGER.debug(f"Sleeping for {sleep_time_s:0.2f} seconds")
            await asyncio.sleep(sleep_time_s)
    except asyncio.TimeoutError as e:
        # This error is caused by our timeout, not by Triton Inference Server client.
        message = "Timeout while waiting for model"
        _LOGGER.error(message)
        raise PyTritonClientTimeoutError(message) from e
    _LOGGER.debug("Server is ready")


async def asyncio_wait_for_model_status_loaded(
    asyncio_client: aio_clients,
    model_name: str,
    sleep_time_s: float,
    model_version: Optional[str] = None,
):
    """Wait for model status loaded.

    Function runs the following async function to check model status:
    ```python
        asyncio_get_model_state(asyncio_client, model_name, model_version)
    ```
    If it return _ModelState.READY, then another async function can check if model is really ready:
    ```python
        asyncio_client.is_model_ready(model_name)
    ```
    This function uses the above functions to check if model is ready together
    with asyncio.wait_for(...) to limit the time of waiting.

    Function contains while loop with sleep to check model status periodically.

    Args:
        asyncio_client: Triton Inference Server client to use for communication
        model_name: name of the model which configuration we're requesting.
        model_version:
            version of the model which configuration we're requesting.
            If model_version is None configuration of the latest model is returned.
            The latest versions of the model are the numerically greatest version numbers.
        timeout_s: timeout to finish model configuration obtain.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
    """
    model_version = model_version or ""
    model_version_msg = model_version or LATEST_MODEL_VERSION
    _LOGGER.debug(f"Waiting for model {model_name}, {model_version_msg} to be ready")
    try:
        while True:
            _LOGGER.debug(f"Checking if model {model_name} is ready")
            is_model_ready = await asyncio_client.is_model_ready(model_name, model_version)
            if is_model_ready:
                break
            _LOGGER.debug(f"Sleeping for {sleep_time_s} seconds")
            await asyncio.sleep(sleep_time_s)
    except asyncio.TimeoutError as e:
        message = f"Timeout while waiting for model {model_name} state (timeout={sleep_time_s:0.2f})"
        _LOGGER.error(message)
        raise PyTritonClientTimeoutError(message) from e
    _LOGGER.debug(f"Model {model_name}, {model_version_msg} is ready")


async def asyncio_wait_for_model_ready(
    asyncio_client: aio_clients,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
):
    """Wait for Triton Inference Server and deployed on it model readiness.

    Args:
        asyncio_client: Triton Inference Server client to use for communication
        model_name: name of the model which configuration we're requesting.
        model_version:
            version of the model which configuration we're requesting.
            If model_version is None configuration of the latest model is returned.
            The latest versions of the model are the numerically greatest version numbers.
        timeout_s: timeout to finish model configuration obtain.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.

    """
    _LOGGER.debug(f"Waiting for model {model_name} to be ready (timeout={timeout_s:0.2f})")
    sleep_time_s = timeout_s * _DEFAULT_ASYNC_SLEEP_FACTOR_S
    try:
        should_finish_before = time.time() + timeout_s
        await asyncio.wait_for(asyncio_wait_for_server_ready(asyncio_client, sleep_time_s), timeout_s)
        _LOGGER.debug(f"Waiting for model {model_name} to be ready")
        timeout_s = max(0, should_finish_before - time.time())
        await asyncio.wait_for(
            asyncio_wait_for_model_status_loaded(
                asyncio_client, model_name=model_name, model_version=model_version, sleep_time_s=sleep_time_s
            ),
            timeout_s,
        )
    except PyTritonClientModelUnavailableError as e:
        _LOGGER.error(f"Failed to obtain model {model_name} config error {e}")
        raise e
    except asyncio.TimeoutError as e:
        _LOGGER.error(f"Failed to obtain model {model_name} config error {e}")
        raise PyTritonClientTimeoutError(
            f"Timeout while waiting for model {model_name} to be ready (timeout={timeout_s:0.2f})"
        ) from e
    _LOGGER.debug(f"Model {model_name} is ready")
