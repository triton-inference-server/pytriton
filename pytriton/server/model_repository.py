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
"""Triton Model Repository class."""
import pathlib
import shutil
from typing import Optional

from pytriton.exceptions import PyTritonError
from pytriton.utils.workspace import Workspace


class TritonModelRepository:
    """Triton Model Repository class."""

    def __init__(self, path: Optional[pathlib.Path], workspace: Workspace):
        """Create new model store if path for existing not provided.

        Use the existing model repository or create new one in workspace.

        Args:
            path: An optional path to existing model store.
            workspace: Workspace details.
        """
        if path is not None and not path.exists():
            raise PyTritonError("Provided model repository path not exists")

        if path is not None:
            self._path = path
            self._created = False
        else:
            self._path = workspace.path / "model-store"
            self._path.mkdir(parents=True, exist_ok=True)
            self._created = True

    @property
    def path(self) -> pathlib.Path:
        """Return the path to location of model store.

        Returns:
            Path to model repository
        """
        return self._path

    def clean(self) -> None:
        """Clean model repository if temporary one was created."""
        if self._created:
            shutil.rmtree(self._path.as_posix())
