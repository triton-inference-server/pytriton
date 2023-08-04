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
"""Utility module supporting model clients."""
import enum
import inspect
import logging
import socket
import sys
import threading
import time
import urllib.parse
from typing import Optional, Union

import gevent
import tritonclient.grpc
import tritonclient.http
from grpc import RpcError
from tritonclient.utils import InferenceServerException

from pytriton.client.exceptions import (
    PyTritonClientInvalidUrlError,
    PyTritonClientModelUnavailableError,
    PyTritonClientTimeoutError,
)
from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT
from pytriton.model_config.parser import ModelConfigParser

_LOGGER = logging.getLogger(__name__)

_TritonSyncClientType = Union[tritonclient.grpc.InferenceServerClient, tritonclient.http.InferenceServerClient]

_DEFAULT_NETWORK_TIMEOUT_S = 60.0  # 1min
_DEFAULT_WAIT_FOR_SERVER_READY_TIMEOUT_S = 60.0  # 1min
_DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S = 300.0  # 5min


class _ModelState(enum.Enum):
    """Describe model state in Triton.

    Attributes:
        LOADING: Loading of model
        UNLOADING: Unloading of model
        UNAVAILABLE: Model is missing or could not be loaded
        READY: Model is ready for inference
    """

    LOADING = "LOADING"
    UNLOADING = "UNLOADING"
    UNAVAILABLE = "UNAVAILABLE"
    READY = "READY"


def _parse_http_response(models):
    models_states = {}
    _LOGGER.debug("Parsing model repository index entries:")
    for model in models:
        _LOGGER.debug(f"    name={model.get('name')} version={model.get('version')} state={model.get('state')}")
        if not model.get("version"):
            continue

        model_state = _ModelState(model["state"]) if model.get("state") else _ModelState.LOADING
        models_states[(model["name"], model["version"])] = model_state

    return models_states


def _parse_grpc_response(models):
    models_states = {}
    _LOGGER.debug("Parsing model repository index entries:")
    for model in models:
        _LOGGER.debug(f"    name={model.name} version={model.version} state={model.state}")
        if not model.version:
            continue

        model_state = _ModelState(model.state) if model.state else _ModelState.LOADING
        models_states[(model.name, model.version)] = model_state

    return models_states


def get_model_state(
    client: _TritonSyncClientType,
    model_name: str,
    model_version: Optional[str] = None,
) -> _ModelState:
    """Obtains state of the model deployed in Triton Inference Server.

    Args:
        client: Triton Inference Server client to use for communication
        model_name: name of the model which state we're requesting.
        model_version:
            version of the model which state we're requesting.
            If model_version is None state of latest model is returned.
            The latest versions of the model are the numerically greatest version numbers.

    Returns:
        Model state. _ModelState.UNAVAILABLE is returned in case if model with given name and version is not found.

    """
    repository_index = client.get_model_repository_index()
    if isinstance(repository_index, list):
        models_states = _parse_http_response(models=repository_index)
    else:
        models_states = _parse_grpc_response(models=repository_index.models)

    if model_version is None:
        requested_model_states = {
            version: state for (name, version), state in models_states.items() if name == model_name
        }
        if not requested_model_states:
            return _ModelState.UNAVAILABLE
        else:
            requested_model_states = sorted(requested_model_states.items(), key=lambda item: int(item[0]))
            latest_version, latest_version_state = requested_model_states[-1]
            return latest_version_state
    else:
        return models_states.get((model_name, model_version), _ModelState.UNAVAILABLE)


def get_model_config(
    client: _TritonSyncClientType,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: Optional[float] = None,
):
    """Obtain configuration of model deployed on the Triton Inference Server.

    Function waits for server readiness.

    Typical use:

        client = tritonclient.grpc.Client("localhost:8001")
        model_config = get_model_config(client, "MyModel", "1", 60.0)
        model_config = get_model_config(client, "MyModel")

    Args:
        client: Triton Inference Server client to use for communication
        model_name: name of the model which configuration we're requesting.
        model_version:
            version of the model which configuration we're requesting.
            If model_version is None configuration of the latest model is returned.
            The latest versions of the model are the numerically greatest version numbers.
        timeout_s: timeout to finish model configuration obtain. Default value is 300.0 s.

    Returns:
        Configuration of requested model.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
        PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    wait_for_model_ready(client, model_name=model_name, model_version=model_version, timeout_s=timeout_s)

    model_version = model_version or ""

    _LOGGER.debug(f"Obtaining model {model_name} config")
    if isinstance(client, tritonclient.grpc.InferenceServerClient):
        response = client.get_model_config(model_name, model_version, as_json=True)
        model_config = response["config"]
    else:
        model_config = client.get_model_config(model_name, model_version)
    model_config = ModelConfigParser.from_dict(model_config)
    _LOGGER.debug(f"Model config: {model_config}")
    return model_config


def _warn_on_too_big_network_timeout(client: _TritonSyncClientType, timeout_s: float):
    if isinstance(client, tritonclient.http.InferenceServerClient):
        connection_pool = client._client_stub._connection_pool
        network_reldiff_s = (connection_pool.network_timeout - timeout_s) / timeout_s
        connection_reldiff_s = (connection_pool.connection_timeout - timeout_s) / timeout_s
        rtol = 0.001
        if network_reldiff_s > rtol or connection_reldiff_s > rtol:
            _LOGGER.warning(
                "Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. "
                f"network_timeout={connection_pool.network_timeout} "
                f"connection_timeout={connection_pool.connection_timeout} "
                f"timeout_s={timeout_s}"
            )


def wait_for_server_ready(
    client: _TritonSyncClientType,
    timeout_s: Optional[float] = None,
    condition: Optional[threading.Condition] = None,
):
    """Waits for Triton Inference Server to be ready.

    Typical use:

        client = tritonclient.http.Client("localhost:8001")
        wait_for_server_ready(client, timeout_s=600.0)

    Args:
        client: Triton Inference Server client to use for communication
        timeout_s: timeout to server get into readiness state. Default value is 60.0 s.
        condition: condition to use for waiting. If None new condition is created.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
    """
    timeout_s = timeout_s if timeout_s is not None else _DEFAULT_WAIT_FOR_SERVER_READY_TIMEOUT_S
    should_finish_before_s = time.time() + timeout_s
    condition = condition or threading.Condition(threading.RLock())

    _warn_on_too_big_network_timeout(client, timeout_s)

    def _is_server_ready():
        try:
            return client.is_server_ready() and client.is_server_live()
        except InferenceServerException as e:
            _LOGGER.warning(f"Exception while checking server readiness: {e}")
            return False
        except (RpcError, ConnectionError, socket.gaierror):  # GRPC and HTTP clients raises these errors
            return False
        except Exception as e:
            _LOGGER.exception(f"Exception while checking server readiness: {e}")
            raise e

    with condition:
        timeout_s = max(0.0, should_finish_before_s - time.time())
        _LOGGER.debug(f"Waiting for server to be ready (timeout={timeout_s})")
        is_server_ready = False
        while not is_server_ready:
            is_server_ready = condition.wait_for(_is_server_ready, timeout=min(1.0, timeout_s))
            if time.time() >= should_finish_before_s:
                raise PyTritonClientTimeoutError("Waiting for server to be ready timed out.")


def wait_for_model_ready(
    client: _TritonSyncClientType,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: Optional[float] = None,
    condition: Optional[threading.Condition] = None,
):
    """Wait for Triton Inference Server and deployed on it model readiness.

    Typical use:

        client = tritonclient.grpc.Client("localhost:8001")
        wait_for_model(client, "MyModel", "1", timeout_s=600.0)
        wait_for_model(client, "MyModel", timeout_s=60.0)

    Args:
        client: Triton Inference Server client to use for communication.
        model_name: name of the model to wait for readiness.
        model_version:
            version of the model to wait for readiness.
            If model_version is None waiting for latest version of the model.
            The latest versions of the model are the numerically greatest version numbers.
        timeout_s: timeout to server and model get into readiness state. Default value is 300.0 s.
        condition: condition to use for waiting. If None new condition is created.

    Raises:
        PyTritonClientTimeoutError: If server and model are not in readiness state before given timeout.
        PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    model_version_msg = model_version if model_version is not None else "<latest>"
    timeout_s = timeout_s if timeout_s is not None else _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
    should_finish_before_s = time.time() + timeout_s
    condition = condition or threading.Condition(threading.RLock())

    def _is_model_ready():
        model_state = get_model_state(client, model_name, model_version)
        if model_state == _ModelState.UNAVAILABLE:
            raise PyTritonClientModelUnavailableError(f"Model {model_name}/{model_version_msg} is unavailable.")

        return model_state == _ModelState.READY and client.is_model_ready(model_name)

    with condition:
        wait_for_server_ready(client, timeout_s=timeout_s, condition=condition)
        timeout_s = max(0.0, should_finish_before_s - time.time())
        _LOGGER.debug(f"Waiting for model {model_name}/{model_version_msg} to be ready (timeout={timeout_s})")
        is_model_ready = False
        while not is_model_ready:
            is_model_ready = condition.wait_for(_is_model_ready, timeout=min(1.0, timeout_s))
            if time.time() >= should_finish_before_s:
                raise PyTritonClientTimeoutError(
                    f"Waiting for model {model_name}/{model_version_msg} to be ready timed out."
                )


def create_client_from_url(url: str, network_timeout_s: Optional[float] = None) -> _TritonSyncClientType:  # type: ignore
    """Create Triton Inference Server client.

    Args:
        url: url of the server to connect to.
            If url doesn't contain scheme (e.g. "localhost:8001") http scheme is added.
            If url doesn't contain port (e.g. "localhost") default port for given scheme is added.
        network_timeout_s: timeout for client commands. Default value is 60.0 s.

    Returns:
        Triton Inference Server client.

    Raises:
        PyTritonClientInvalidUrlError: If provided Triton Inference Server url is invalid.
    """
    if not isinstance(url, str):
        raise PyTritonClientInvalidUrlError(f"Invalid url {url}. Url must be a string.")

    try:
        parsed_url = urllib.parse.urlparse(url)
        # change in py3.9+
        # https://github.com/python/cpython/commit/5a88d50ff013a64fbdb25b877c87644a9034c969
        if sys.version_info < (3, 9) and not parsed_url.scheme and "://" in parsed_url.path:
            raise ValueError(f"Invalid url {url}. Only grpc and http are supported.")
        if (sys.version_info < (3, 9) and not parsed_url.scheme and "://" not in parsed_url.path) or (
            sys.version_info >= (3, 9) and parsed_url.scheme and not parsed_url.netloc
        ):
            _LOGGER.debug(f"Adding http scheme to {url}")
            parsed_url = urllib.parse.urlparse(f"http://{url}")

        scheme = parsed_url.scheme.lower()
        if scheme not in ["grpc", "http"]:
            raise ValueError(f"Invalid scheme {scheme}. Only grpc and http are supported.")
        port = parsed_url.port or {"grpc": DEFAULT_GRPC_PORT, "http": DEFAULT_HTTP_PORT}[scheme]
    except ValueError as e:
        raise PyTritonClientInvalidUrlError(f"Invalid url {url}") from e

    url = f"{parsed_url.hostname}:{port}"
    triton_client_lib = {"grpc": tritonclient.grpc, "http": tritonclient.http}[scheme]

    def _monkey_patch_client():
        old_del = getattr(triton_client_lib.InferenceServerClient, "__del__")  # noqa: B009

        # Monkey patch __del__ method from client to catch error in client when instance is garbage collected.
        # This is needed because we are closing client in __exit__ method or in close method.
        # (InferenceClient uses gevent library which does not support closing twice from different threads)
        def _monkey_patched_del(self):
            """Monkey patched del."""
            try:
                old_del(self)
            except gevent.exceptions.InvalidThreadUseError:
                _LOGGER.warning("gevent.exceptions.InvalidThreadUseError in __del__ of InferenceServerClient")
            except Exception as e:
                _LOGGER.error("Exception in __del__ of InferenceServerClient: %s", e)

        if old_del:
            triton_client_lib.InferenceServerClient.__del__ = _monkey_patched_del

    _monkey_patch_client()

    if scheme == "grpc":
        # by default grpc client has very large number of timeout, thus we want to make it equal to http client timeout
        network_timeout_s = _DEFAULT_NETWORK_TIMEOUT_S if network_timeout_s is None else network_timeout_s
        _LOGGER.warning(
            f"tritonclient.grpc doesn't support timeout for other commands than infer. Ignoring network_timeout: {network_timeout_s}."
        )

    triton_client_init_kwargs = {}
    if network_timeout_s is not None:
        triton_client_init_kwargs.update(
            **{
                "grpc": {},
                # connection_timeout
                #     This value sets the maximum time allowed for establishing a connection to the server.
                #     We use the inference timeout here instead of the init timeout because the init timeout
                #     is meant for waiting for the model to be ready. The connection timeout should be shorter
                #     than the init timeout because it only checks if connection is established (e.g. correct port)
                # network_timeout
                #     This value sets the maximum time allowed for each network request
                #     in both model loading and inference process
                "http": {"connection_timeout": network_timeout_s, "network_timeout": network_timeout_s},
            }[scheme]
        )

    _LOGGER.debug(f"Creating InferenceServerClient for {parsed_url.scheme}://{url} with {triton_client_init_kwargs}")
    return triton_client_lib.InferenceServerClient(url, **triton_client_init_kwargs)


def get_client_lib_from_client(client: _TritonSyncClientType):
    """Get Triton Inference Server client library module.

    Args:
        client: Triton Inference Server client.

    Returns:
        Triton Inference Server client library module.
    """
    package_name: str = inspect.getmodule(client).__package__  # pytype: disable=attribute-error
    return sys.modules[package_name]
