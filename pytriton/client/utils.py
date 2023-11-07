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
import dataclasses
import enum
import logging
import socket
import sys
import time
import urllib
import warnings
from typing import Optional, Union

import tritonclient.grpc
import tritonclient.http
import tritonclient.http.aio
from grpc import RpcError
from tritonclient.utils import InferenceServerException

from pytriton.client.exceptions import PyTritonClientInvalidUrlError, PyTritonClientTimeoutError
from pytriton.client.warnings import NotSupportedTimeoutWarning
from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT
from pytriton.model_config.parser import ModelConfigParser

_LOGGER = logging.getLogger(__name__)

_TritonSyncClientType = Union[tritonclient.grpc.InferenceServerClient, tritonclient.http.InferenceServerClient]

_DEFAULT_NETWORK_TIMEOUT_S = 60.0  # 1min
_DEFAULT_WAIT_FOR_SERVER_READY_TIMEOUT_S = 60.0  # 1min
_DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S = 300.0  # 5min

LATEST_MODEL_VERSION = "<latest>"


# Special value for model_version argument. If model_version is None, the latest version of the model is returned.


class ModelState(enum.Enum):
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


def parse_http_response(models):
    """Parse model repository index response from Triton Inference Server for HTTP."""
    models_states = {}
    _LOGGER.debug("Parsing model repository index entries:")
    for model in models:
        _LOGGER.debug(f"    name={model.get('name')} version={model.get('version')} state={model.get('state')}")
        if not model.get("version"):
            continue

        model_state = ModelState(model["state"]) if model.get("state") else ModelState.LOADING
        models_states[(model["name"], model["version"])] = model_state

    return models_states


def parse_grpc_response(models):
    """Parse model repository index response from Triton Inference Server for GRCP."""
    models_states = {}
    _LOGGER.debug("Parsing model repository index entries:")
    for model in models:
        _LOGGER.debug(f"    name={model.name} version={model.version} state={model.state}")
        if not model.version:
            continue

        model_state = ModelState(model.state) if model.state else ModelState.LOADING
        models_states[(model.name, model.version)] = model_state

    return models_states


def get_model_state(
    client: _TritonSyncClientType,
    model_name: str,
    model_version: Optional[str] = None,
) -> ModelState:
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
            return latest_version_state
    else:
        state = models_states.get((model_name, model_version), ModelState.UNAVAILABLE)
        return state


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
            warnings.warn(
                "Client network and/or connection timeout is smaller than requested timeout_s. This may cause unexpected behavior. "
                f"network_timeout={connection_pool.network_timeout} "
                f"connection_timeout={connection_pool.connection_timeout} "
                f"timeout_s={timeout_s}",
                NotSupportedTimeoutWarning,
                stacklevel=1,
            )


def wait_for_server_ready(
    client: _TritonSyncClientType,
    timeout_s: Optional[float] = None,
):
    """Waits for Triton Inference Server to be ready.

    Typical use:

        client = tritonclient.http.Client("localhost:8001")
        wait_for_server_ready(client, timeout_s=600.0)

    Args:
        client: Triton Inference Server client to use for communication
        timeout_s: timeout to server get into readiness state. Default value is 60.0 s.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
    """
    timeout_s = timeout_s if timeout_s is not None else _DEFAULT_WAIT_FOR_SERVER_READY_TIMEOUT_S
    should_finish_before_s = time.time() + timeout_s
    _warn_on_too_big_network_timeout(client, timeout_s)

    def _is_server_ready():
        try:
            return client.is_server_ready() and client.is_server_live()
        except InferenceServerException:
            return False
        except (RpcError, ConnectionError, socket.gaierror):  # GRPC and HTTP clients raises these errors
            return False
        except Exception as e:
            _LOGGER.exception(f"Exception while checking server readiness: {e}")
            raise e

    timeout_s = max(0.0, should_finish_before_s - time.time())
    _LOGGER.debug(f"Waiting for server to be ready (timeout={timeout_s})")
    is_server_ready = _is_server_ready()
    while not is_server_ready:
        time.sleep(min(1.0, timeout_s))
        is_server_ready = _is_server_ready()
        if not is_server_ready and time.time() >= should_finish_before_s:
            raise PyTritonClientTimeoutError("Waiting for server to be ready timed out.")


def wait_for_model_ready(
    client: _TritonSyncClientType,
    model_name: str,
    model_version: Optional[str] = None,
    timeout_s: Optional[float] = None,
):
    """Wait for Triton Inference Server to be ready.

    Args:
        client: Triton Inference Server client to use for communication.
        model_name: name of the model to wait for readiness.
        model_version:
            version of the model to wait for readiness.
            If model_version is None waiting for latest version of the model.
            The latest versions of the model are the numerically greatest version numbers.
        timeout_s: timeout to server and model get into readiness state. Default value is 300.0 s.

    Raises:
        PyTritonClientTimeoutError: If server readiness didn't finish before given timeout.
    """
    model_version = model_version or ""
    model_version_msg = model_version or LATEST_MODEL_VERSION
    timeout_s = timeout_s if timeout_s is not None else _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
    should_finish_before_s = time.time() + timeout_s

    wait_for_server_ready(client, timeout_s=timeout_s)
    timeout_s = max(0.0, should_finish_before_s - time.time())
    _LOGGER.debug(f"Waiting for model {model_name}/{model_version_msg} to be ready (timeout={timeout_s})")
    is_model_ready = client.is_model_ready(model_name, model_version)
    while not is_model_ready:
        time.sleep(min(1.0, timeout_s))
        is_model_ready = client.is_model_ready(model_name, model_version)

        if not is_model_ready and time.time() >= should_finish_before_s:
            raise PyTritonClientTimeoutError(
                f"Waiting for model {model_name}/{model_version_msg} to be ready timed out."
            )


def create_client_from_url(
    url: str, network_timeout_s: Optional[float] = None
) -> _TritonSyncClientType:  # type: ignore
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
    url = TritonUrl.from_url(url)
    triton_client_lib = {"grpc": tritonclient.grpc, "http": tritonclient.http}[url.scheme]

    if url.scheme == "grpc":
        # by default grpc client has very large number of timeout, thus we want to make it equal to http client timeout
        network_timeout_s = _DEFAULT_NETWORK_TIMEOUT_S if network_timeout_s is None else network_timeout_s
        warnings.warn(
            f"tritonclient.grpc doesn't support timeout for other commands than infer. Ignoring network_timeout: {network_timeout_s}.",
            NotSupportedTimeoutWarning,
            stacklevel=1,
        )

    triton_client_init_kwargs = {}
    if network_timeout_s is not None:
        triton_client_init_kwargs.update(
            **{
                "grpc": {},
                "http": {"connection_timeout": network_timeout_s, "network_timeout": network_timeout_s},
            }[url.scheme]
        )

    _LOGGER.debug(f"Creating InferenceServerClient for {url.with_scheme} with {triton_client_init_kwargs}")
    return triton_client_lib.InferenceServerClient(url.without_scheme, **triton_client_init_kwargs)


@dataclasses.dataclass
class TritonUrl:
    """TritonUrl class for parsing Triton Inference Server url.

    Attributes:
        scheme: scheme of the url (http or grpc)
        hostname: hostname of the url
        port: port of the url

    Examples:
        triton_url = TritonUrl.from_url("localhost:8000")
        triton_url.with_scheme
        >>> "http://localhost:8000"
        triton_url.without_scheme
        >>> "localhost:8000"
        triton_url.scheme, triton_url.hostname, triton_url.port
        >>> ("http", "localhost", 8000)
    """

    scheme: str
    hostname: str
    port: int

    @classmethod
    def from_url(cls, url):
        """Parse triton url and create TritonUrl instance.

        Returns:
            TritonUrl object with scheme, hostname and port.
        """
        if not isinstance(url, str):
            raise PyTritonClientInvalidUrlError(f"Invalid url {url}. Url must be a string.")
        try:
            parsed_url = urllib.parse.urlparse(url)
            # change in py3.9+
            # https://github.com/python/cpython/commit/5a88d50ff013a64fbdb25b877c87644a9034c969
            if sys.version_info < (3, 9) and not parsed_url.scheme and "://" in parsed_url.path:
                raise ValueError(f"Invalid url {url}. Only grpc and http are supported.")
            if (not parsed_url.scheme and "://" not in parsed_url.path) or (
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
        return cls(scheme, parsed_url.hostname, port)

    @property
    def with_scheme(self):
        """Get Triton Inference Server url with scheme."""
        return f"{self.scheme}://{self.hostname}:{self.port}"

    @property
    def without_scheme(self):
        """Get Triton Inference Server url without scheme."""
        return f"{self.hostname}:{self.port}"
