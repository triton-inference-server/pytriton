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

import numpy as np
from tritonclient.grpc import InferenceServerException

from pytriton.client import ModelClient
from pytriton.client.utils import create_client_from_url, wait_for_server_ready
from pytriton.constants import CREATE_TRITON_CLIENT_TIMEOUT_S, DEFAULT_TRITON_STARTUP_TIMEOUT_S
from pytriton.exceptions import PyTritonInvalidOperationError
from pytriton.model_config.common import generate_warmup_requests
from pytriton.models.model import Model
from pytriton.proxy.types import Request

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

        This method performs warmup in Python before loading the model in Triton.
        """
        LOGGER.debug("Loading model %s with version %s with warmup support.", model.model_name, model.model_version)

        # Check if model has warmup configured
        has_warmup = model.config.model_warmup is not None and len(model.config.model_warmup) > 0

        if has_warmup and not local_model_store:
            LOGGER.debug("Model %s has warmup configured - performing pre-Triton warmup", model.model_name)
            self._perform_pre_triton_warmup(model)

        # Load model normally (without warmup config in Triton)
        self._load_model(model, local_model_store)
        model.setup()

    def _perform_pre_triton_warmup(self, model: Model):
        """Perform warmup by calling the inference callable directly with numpy data.

        This method:
        1. Generates warmup data from model warmup configuration
        2. Calls the inference callable directly with the warmup data
        3. This warms up the model before Triton starts loading it
        """
        LOGGER.info("Starting pre-Triton warmup for model %s", model.model_name)

        if not model.config.model_warmup:
            LOGGER.debug("No warmup configuration found for model %s", model.model_name)
            return

        # Get the processed inputs with auto-generated names from Triton model config
        triton_config = model._get_triton_model_config()
        processed_inputs = triton_config.inputs or []
        processed_outputs = triton_config.outputs or []

        # Generate warmup requests from configuration
        try:
            warmup_requests = generate_warmup_requests(
                model.config.model_warmup,
                processed_inputs,  # Use processed inputs with auto-generated names
                processed_outputs,
            )

            if not warmup_requests:
                LOGGER.debug("No warmup requests generated for model %s", model.model_name)
                return

            LOGGER.info("Generated %d warmup requests for model %s", len(warmup_requests), model.model_name)

            # Perform warmup by calling each inference function directly
            for i, infer_function in enumerate(model.infer_functions):
                LOGGER.debug("Warming up inference function %d for model %s", i, model.model_name)

                for j, warmup_data in enumerate(warmup_requests):
                    LOGGER.debug(
                        "Running warmup request %d/%d for inference function %d", j + 1, len(warmup_requests), i
                    )

                    try:
                        # Call the inference function with the correct interface
                        result = infer_function([Request(data=warmup_data)])
                        LOGGER.debug("Warmup request %d completed successfully for inference function %d", j + 1, i)
                    except Exception as e:
                        LOGGER.warning("Warmup request %d failed for inference function %d: %s", j + 1, i, str(e))
                        # Continue with other warmup requests even if one fails
                        continue

            LOGGER.info("Pre-Triton warmup completed for model %s", model.model_name)

        except Exception as e:
            LOGGER.error("Failed to perform pre-Triton warmup for model %s: %s", model.model_name, str(e))
            # Don't raise the exception - warmup failure shouldn't prevent model loading
            LOGGER.warning("Continuing with model loading despite warmup failure")

    def _call_inference_function(self, infer_function, warmup_data: Dict[str, np.ndarray]):
        """Call the inference function with the correct interface.

        This method handles both decorated and undecorated inference functions by
        trying the @batch interface first and falling back to Request objects.

        Args:
            infer_function: The inference function to call
            warmup_data: Dictionary mapping input names to numpy arrays

        Returns:
            The result from the inference function
        """
        # Try calling with @batch decorator interface first (list of dict-like objects)
        try:
            requests = [warmup_data]  # warmup_data is already a dict with input names -> numpy arrays
            result = infer_function(requests)
            LOGGER.debug("Successfully called inference function with @batch interface")
            return result
        except Exception as batch_error:
            LOGGER.debug("@batch interface failed, trying Request interface: %s", str(batch_error))

            # Fall back to undecorated function interface (List[Request] objects)
            try:
                request = Request(data=warmup_data)
                requests = [request]
                result = infer_function(requests)
                LOGGER.debug("Successfully called inference function with Request interface")
                return result
            except Exception as request_error:
                LOGGER.error(
                    "Both @batch and Request interfaces failed. @batch error: %s, Request error: %s",
                    str(batch_error),
                    str(request_error),
                )
                raise request_error  # Raise the last error

    def _is_function_decorated_with_batch(self, func):
        """Check if function is decorated with @batch decorator.

        Args:
            func: Function to check

        Returns:
            True if function is decorated with @batch, False otherwise
        """
        from pytriton.decorators import _get_wrapt_stack, batch

        try:
            # Get the stack of wrappers applied to this function
            wrapper_stack = _get_wrapt_stack(func)

            # Check if any wrapper in the stack is the batch decorator
            for wrapped_with_wrapper in wrapper_stack:
                if wrapped_with_wrapper.wrapper is batch:
                    return True

            return False
        except Exception:
            # If we can't determine the wrapper stack, assume it's not batch decorated
            return False

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
