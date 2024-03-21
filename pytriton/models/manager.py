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
"""ModelManager class.

The ModelManager is responsible for maintaining the models that has to be server on Triton Inference Server.

    Examples of use:
        manager = ModelManager(model_repository)
        manager.add_model(model)

        manager.create_models()
"""

import contextlib
import json
import logging
import pathlib
import socket
from typing import Dict, Iterable, Optional, Tuple

from tritonclient.grpc import InferenceServerException

from pytriton.client import ModelClient
from pytriton.client.utils import create_client_from_url, wait_for_server_ready
from pytriton.constants import CREATE_TRITON_CLIENT_TIMEOUT_S, DEFAULT_TRITON_STARTUP_TIMEOUT_S
from pytriton.exceptions import PyTritonInvalidOperationError
from pytriton.models.model import Model

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """ModelManager class for maintaining Triton models."""

    def __init__(
        self,
        triton_url: str,
        model_store_path: Optional[pathlib.Path] = None,
    ):
        """Create ModelManager object.

        Args:
            triton_url: Triton server URL
            model_store_path: Path to local model store
        """
        self._triton_url = triton_url
        self._models: Dict[Tuple[str, int], Model] = {}
        self._model_store_path = model_store_path

    @property
    def models(self) -> Iterable[Model]:
        """List models added to manage.

        Returns:
            List with models added to ModelManager.
        """
        return self._models.values()

    def add_model(self, model: Model, load_model: bool = False) -> None:
        """Add model to manage.

        Args:
            model: Model instance
            load_model: If True, model will be loaded to Triton server.
        """
        key = self._format_key(model)
        if key in self._models:
            raise PyTritonInvalidOperationError("Cannot add model with the same name twice.")

        LOGGER.debug(f"Adding {model.model_name} ({model.model_version}) to registry under {key}.")
        self._models[key] = model

        _is_model_store_local = self._model_store_path is not None
        if _is_model_store_local:
            model.generate_model(self._model_store_path)

        if load_model:
            self._load_model(model, _is_model_store_local)
            model.setup()

    def load_models(self) -> None:
        """Load bound models to Triton server and setup loaded models."""
        for model in self._models.values():
            if not model.is_alive():
                self._load_model(model)
                model.setup()

    def setup_models(self) -> None:
        """Setup loaded models."""
        for model in self._models.values():
            if not model.is_alive():
                model.setup()

    def clean(self) -> None:
        """Clean the model and internal registry."""
        with contextlib.closing(
            create_client_from_url(self._triton_url, network_timeout_s=CREATE_TRITON_CLIENT_TIMEOUT_S)
        ) as client:
            server_live = False
            try:
                server_live = client.is_server_live()
            # TimeoutError and ConnectionRefusedError are derived from OSError so they are redundant here
            # OSError is raised from gevent/_socketcommon.py:590 sometimes, when server is not ready
            except (socket.timeout, OSError, InferenceServerException):
                pass
            except Exception as ex:
                LOGGER.error(f"Unexpected exception during server live check: {ex}")
                raise ex

            for name, model in self._models.items():
                LOGGER.debug(f"Clean model {name}.")
                model.clean()
                if server_live:
                    client.unload_model(model.model_name)

            if server_live:
                # after unload there is a short period of time when server is not ready
                wait_for_server_ready(client, timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S)

        self._models.clear()

    def _format_key(self, model: Model) -> Tuple[str, int]:
        key = (model.model_name.lower(), model.model_version)
        return key

    def _load_model(self, model: Model, local_model_store=False):
        """Prepare model config and required files dict and load model to Triton server."""
        LOGGER.debug(f"Creating model {model.model_name} with version {model.model_version}.")
        config = None if local_model_store else json.dumps(model.get_model_config())
        files = None if local_model_store else model.get_proxy_model_files()
        with ModelClient(
            url=self._triton_url, model_name=model.model_name, model_version=str(model.model_version)
        ) as client:
            client.wait_for_server(timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S)
            client.load_model(config=config, files=files)
        LOGGER.debug("Done.")
