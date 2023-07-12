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
"""Utility module supporting model clients."""
import enum
import logging
import threading
import time
from socket import gaierror
from typing import Optional, Union

import tritonclient.grpc
import tritonclient.http
from grpc import RpcError
from tritonclient.utils import InferenceServerException

from pytriton.client.exceptions import PyTritonClientModelUnavailableError, PyTritonClientTimeoutError
from pytriton.model_config.parser import ModelConfigParser

_LOGGER = logging.getLogger(__name__)

_TritonSyncClientType = Union[tritonclient.grpc.InferenceServerClient, tritonclient.http.InferenceServerClient]

_DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S = 300  # 5min
# Tritonclient doesn't support timeout yet
# _DEFAULT_GRPC_NETWORK_TIMEOUT_S = 60  # 1min


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
    # Tritonclient doesn't support timeout yet
    # grpc_network_timeout_s: float = _DEFAULT_GRPC_NETWORK_TIMEOUT_S,
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
    # Timeout flow is quite complex. Please see ModelClient.__init__ for more details.
    if isinstance(client, tritonclient.grpc.InferenceServerClient):
        # For the GRPC protocol, the timeout is passed to the infer method as client_timeout
        # This timeout applies to the whole inference process and each network request
        # Tritonclient doesn't support timeout for get_model_repository_index yet
        repository_index = client.get_model_repository_index()
    else:
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
    init_timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
    # Tritonclient doesn't support timeout for get_model_repository_index yet
    # grpc_network_timeout_s: float = _DEFAULT_GRPC_NETWORK_TIMEOUT_S,
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
        timeout_s: timeout to finish model configuration obtain.

    Returns:
        Configuration of requested model.

    Raises:
        PyTritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
        PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    wait_for_model_ready(
        client,
        model_name=model_name,
        model_version=model_version,
        init_timeout_s=init_timeout_s,
        # Tritonclient doesn't support timeout yet
        # grpc_network_timeout_s=grpc_network_timeout_s,
    )

    model_version = model_version or ""

    _LOGGER.debug(f"Obtaining model {model_name} config")
    # Timeout flow is quite complex. Please see ModelClient.__init__ for more details.
    if isinstance(client, tritonclient.grpc.InferenceServerClient):
        # For the GRPC protocol, the timeout is passed to the infer method as client_timeout
        # This timeout applies to the whole inference process and each network request
        # Tritonclient doesn't support timeout for get_model_config
        response = client.get_model_config(
            model_name,
            model_version,
            as_json=True,
        )
        model_config = response["config"]
    else:
        model_config = client.get_model_config(model_name, model_version)
    model_config = ModelConfigParser.from_dict(model_config)
    _LOGGER.debug(f"Model config: {model_config}")
    return model_config


def wait_for_model_ready(
    client: _TritonSyncClientType,
    model_name: str,
    model_version: Optional[str] = None,
    init_timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
    # Tritonclient doesn't support timeout yet
    # grpc_network_timeout_s: float = _DEFAULT_GRPC_NETWORK_TIMEOUT_S,
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
        timeout_s: timeout to server and model get into readiness state.

    Raises:
        PyTritonClientTimeoutError: If server and model are not in readiness state before given timeout.
        PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    model_version_msg = model_version if model_version is not None else "<latest>"
    should_finish_before_s = time.time() + init_timeout_s
    condition = threading.Condition()

    def _is_server_ready():
        try:
            # Timeout flow is quite complex. Please see ModelClient.__init__ for more details.
            if isinstance(client, tritonclient.grpc.InferenceServerClient):
                # For the GRPC protocol, the timeout is passed to the infer method as client_timeout
                # This timeout applies to the whole inference process and each network request
                #
                # Tritonclient doesn't support timeout for is_server_ready and is_server_live
                return client.is_server_ready() and client.is_server_live()
                # client._client_stub.ServerReady fails with error:
                # > Exception while checking server readiness: <_InactiveRpcError of RPC that terminated with:
                # > status = StatusCode.UNAVAILABLE
                # > details = "failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8001: Failed to connect to remote host: Connection refused"
                # > debug_error_string = "UNKNOWN:failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:8001: Failed to connect to remote host: Connection refused {created_time:"2023-07-05T14:52:02.138173886+02:00", grpc_status:14}"
                # return client._client_stub.ServerReady(request=GRPCServerReadyRequest(), timeout=grpc_network_timeout_s).ready and client._client_stub.ServerLive(request=GRPCServerLiveRequest(), timeout=grpc_network_timeout_s).live
            else:
                return client.is_server_ready() and client.is_server_live()
        except InferenceServerException as e:
            _LOGGER.warning(f"Exception while checking server readiness: {e}")
            return False
        except RpcError as e:  # GRPC stub throws this exception when network timeouts
            _LOGGER.warning(f"Exception while checking server readiness: {e}")
            return False
        except gaierror as e:  # HTTP client raises this error
            _LOGGER.warning(f"Exception while checking server readiness: {e}")
            return False
        except ConnectionRefusedError as e:  # HTTP client raises this error
            _LOGGER.warning(f"Exception while checking server readiness: {e}")
            return False
        except Exception as e:
            _LOGGER.exception(f"Exception while checking server readiness: {e}")
            raise e

    def _is_model_ready():
        model_state = get_model_state(
            client,
            model_name,
            model_version,
            # Tritonclient doesn't support timeout yet
            # grpc_network_timeout_s=grpc_network_timeout_s,
        )
        if model_state == _ModelState.UNAVAILABLE:
            raise PyTritonClientModelUnavailableError(f"Model {model_name}/{model_version_msg} is unavailable.")

        return model_state == _ModelState.READY and client.is_model_ready(model_name)

    with condition:
        timeout_s = max(0.0, should_finish_before_s - time.time())
        _LOGGER.debug(f"Waiting for server to be ready (timeout={timeout_s})")
        is_server_ready = False
        while not is_server_ready:
            is_server_ready = condition.wait_for(_is_server_ready, timeout=min(1.0, timeout_s))
            if time.time() >= should_finish_before_s:
                raise PyTritonClientTimeoutError("Waiting for server to be ready timed out.")

        timeout_s = max(0.0, should_finish_before_s - time.time())
        _LOGGER.debug(f"Waiting for model {model_name}/{model_version_msg} to be ready (timeout={timeout_s})")
        is_model_ready = False
        while not is_model_ready:
            is_model_ready = condition.wait_for(_is_model_ready, timeout=min(1.0, timeout_s))
            if time.time() >= should_finish_before_s:
                raise PyTritonClientTimeoutError(
                    f"Waiting for model {model_name}/{model_version_msg} to be ready timed out."
                )
