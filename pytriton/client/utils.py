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
from typing import Optional, Union

import tritonclient.grpc
import tritonclient.http

from pytriton.client.exceptions import PytritonClientModelUnavailableError, PytritonClientTimeoutError
from pytriton.model_config.parser import ModelConfigParser

_LOGGER = logging.getLogger(__name__)

_TritonSyncClientType = Union[tritonclient.grpc.InferenceServerClient, tritonclient.http.InferenceServerClient]

_DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S = 300  # 5min


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
    timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
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
        PytritonClientTimeoutError: If obtain of model configuration didn't finish before given timeout.
        PytritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    should_finish_before_s = time.time() + timeout_s
    wait_for_model_ready(client, model_name=model_name, model_version=model_version, timeout_s=timeout_s)

    model_version = model_version or ""

    timeout_s = max(0.0, should_finish_before_s - time.time())
    _LOGGER.debug(f"Obtaining model {model_name} config (timeout={timeout_s:0.2f})")
    if isinstance(client, tritonclient.grpc.InferenceServerClient):
        response = client.get_model_config(model_name, model_version, as_json=True)
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
    timeout_s: float = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
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
        PytritonClientTimeoutError: If server and model are not in readiness state before given timeout.
        PytritonClientModelUnavailableError: If model with given name (and version) is unavailable.
    """
    model_version_msg = model_version if model_version is not None else "<latest>"
    should_finish_before_s = time.time() + timeout_s
    condition = threading.Condition()

    def _is_server_ready():
        try:
            return client.is_server_ready() and client.is_server_live()
        except Exception:
            return False

    def _is_model_ready():
        model_state = get_model_state(client, model_name, model_version)
        if model_state == _ModelState.UNAVAILABLE:
            raise PytritonClientModelUnavailableError(f"Model {model_name}/{model_version_msg} is unavailable.")

        return model_state == _ModelState.READY and client.is_model_ready(model_name)

    with condition:
        timeout_s = max(0.0, should_finish_before_s - time.time())
        _LOGGER.debug(f"Waiting for server to be ready (timeout={timeout_s})")
        is_server_ready = False
        while not is_server_ready:
            is_server_ready = condition.wait_for(_is_server_ready, timeout=min(1.0, timeout_s))
            if time.time() >= should_finish_before_s:
                raise PytritonClientTimeoutError("Waiting for server to be ready timed out.")

        timeout_s = max(0.0, should_finish_before_s - time.time())
        _LOGGER.debug(f"Waiting for model {model_name}/{model_version_msg} to be ready (timeout={timeout_s})")
        is_model_ready = False
        while not is_model_ready:
            is_model_ready = condition.wait_for(_is_model_ready, timeout=min(1.0, timeout_s))
            if time.time() >= should_finish_before_s:
                raise PytritonClientTimeoutError(
                    f"Waiting for model {model_name}/{model_version_msg} to be ready timed out."
                )
