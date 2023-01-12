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
"""ModelManager class.

The ModelManager is responsible for maintaining the models that has to be server on Triton Inference Server.

    Examples of use:
        manager = ModelManager(model_repository)
        manager.add_model(model)

        manager.create_models()
"""
import logging
from typing import Dict, Iterable, Tuple

from pytriton.exceptions import PytritonInvalidOperationError
from pytriton.models.model import Model
from pytriton.server.model_repository import TritonModelRepository

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """ModelManager class for maintaining Triton models."""

    def __init__(self, model_repository: TritonModelRepository):
        """Create ModelManager object.

        Args:
            model_repository: Triton model repository object
        """
        self._model_repository = model_repository
        self._models: Dict[Tuple[str, int], Model] = {}

    @property
    def models(self) -> Iterable[Model]:
        """List models added to manage.

        Returns:
            List with models added to ModelManager.
        """
        return self._models.values()

    def add_model(self, model: Model) -> None:
        """Add model to manage.

        Args:
            model: Model instance
        """
        key = self._format_key(model)
        if key in self._models:
            raise PytritonInvalidOperationError("Cannot add model with the same name twice.")

        LOGGER.debug(f"Adding {model.model_name} ({model.model_version}) to registry under {key}.")
        self._models[key] = model

    def create_models(self) -> None:
        """Create models in model repository and setup necessary dependencies."""
        for model in self._models.values():
            LOGGER.debug(f"Crating model {model.model_name} with version {model.model_version}.")
            model.generate_model(self._model_repository.path)
            model.setup()
            LOGGER.debug("Done.")

    def clean(self) -> None:
        """Clean the model and internal registry."""
        for name, model in self._models.items():
            LOGGER.debug(f"Clean model {name}.")
            model.clean()

        self._models.clear()

    def _format_key(self, model: Model) -> Tuple[str, int]:
        key = (model.model_name.lower(), model.model_version)
        return key
