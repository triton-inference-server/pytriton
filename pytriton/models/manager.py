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
import time
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

        LOGGER.debug("Adding %s (%s) to registry under %s.", model.model_name, model.model_version, key)
        self._models[key] = model

        _is_model_store_local = self._model_store_path is not None
        if _is_model_store_local:
            model.generate_model(self._model_store_path)

        if load_model:
            self._load_model_with_warmup_support(model, _is_model_store_local)

    def load_models(self) -> None:
        """Load bound models to Triton server and setup loaded models."""
        for model in self._models.values():
            if not model.is_alive():
                self._load_model_with_warmup_support(model)

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
                LOGGER.error("Unexpected exception during server live check: %s", ex)
                raise ex

            for name, model in self._models.items():
                LOGGER.debug("Clean model %s.", name)
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

    def _load_model_with_warmup_support(self, model: Model, local_model_store=False):
        """Load model to Triton server with proper warmup support.
        
        This method establishes the proxy connection before warmup runs to ensure
        warmup can work properly when configured.
        """
        LOGGER.debug("Loading model %s with version %s with warmup support.", model.model_name, model.model_version)
        
        # Check if model has warmup configured
        model_config_dict = model.get_model_config()
        has_warmup = model_config_dict.get("model_warmup") is not None
        
        if has_warmup and not local_model_store:
            LOGGER.debug("Model %s has warmup configured - using warmup-safe loading sequence", model.model_name)
            self._load_model_with_delayed_warmup(model, local_model_store)
        else:
            # Use original loading sequence for models without warmup or local model store
            self._load_model(model, local_model_store)
            model.setup()

    def _load_model_with_delayed_warmup(self, model: Model, local_model_store=False):
        """Load model with delayed warmup to ensure proxy connection is established first.
        
        This method:
        1. Temporarily removes warmup from model config
        2. Loads model without warmup (establishes backend)
        3. Sets up proxy connection
        4. Reloads model with warmup enabled (warmup now works)
        """
        # Get the original model config
        original_config_dict = model.get_model_config()
        warmup_config = original_config_dict.get("model_warmup")
        
        # Create a temporary config without warmup
        temp_config_dict = original_config_dict.copy()
        temp_config_dict.pop("model_warmup", None)
        
        config_without_warmup = json.dumps(temp_config_dict)
        files = model.get_proxy_model_files()
        
        with ModelClient(
            url=self._triton_url, model_name=model.model_name, model_version=str(model.model_version)
        ) as client:
            client.wait_for_server(timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S)
            
            # Step 1: Load model without warmup to initialize backend
            LOGGER.debug("Loading model %s without warmup to initialize backend", model.model_name)
            client.load_model(config=config_without_warmup, files=files)
            
            # Step 2: Wait a moment for backend to be fully initialized
            time.sleep(1.0)
            
            # Step 3: Setup proxy connection now that backend is running
            LOGGER.debug("Setting up proxy connection for model %s", model.model_name)
            model.setup()
            
            # Step 4: Reset proxy connections before unloading
            LOGGER.debug("Resetting proxy connections before reload for model %s", model.model_name)
            model.reset_proxy_connections()
            
            # Step 5: Unload model
            LOGGER.debug("Unloading model %s before warmup reload", model.model_name)
            client.unload_model(model.model_name)
            
            # Wait for unload to complete
            time.sleep(0.5)
            
            # Step 6: Reload with the original config including warmup
            LOGGER.debug("Reloading model %s with warmup enabled", model.model_name)
            original_config = json.dumps(original_config_dict)
            client.load_model(config=original_config, files=files)
            
            # Step 7: Setup proxy connection again for the reloaded model
            LOGGER.debug("Setting up proxy connection for reloaded model %s", model.model_name)
            model.setup()
            
        LOGGER.debug("Model loading with warmup support completed for %s", model.model_name)

    def _load_model(self, model: Model, local_model_store=False):
        """Original model loading method (for backwards compatibility)."""
        LOGGER.debug("Creating model %s with version %s.", model.model_name, model.model_version)
        config = None if local_model_store else json.dumps(model.get_model_config())
        files = None if local_model_store else model.get_proxy_model_files()
        with ModelClient(
            url=self._triton_url, model_name=model.model_name, model_version=str(model.model_version)
        ) as client:
            client.wait_for_server(timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S)
            client.load_model(config=config, files=files)
        LOGGER.debug("Done.")
