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
"""Workspace class for handling space to store artifacts."""
import logging
import pathlib
import shutil
import tempfile
from typing import Optional, Union

LOGGER = logging.getLogger(__name__)


class Workspace:
    """Class for storing the workspace information."""

    def __init__(self, workspace_path: Optional[Union[str, pathlib.Path]] = None):
        """Initialize workspace in the provided path or create workspace in default location.

        Args:
            workspace_path: Path to a directory where workspace has to be created (optional).
                If not provided workspace with random name will be created in ~/.cache/pytriton directory.

        Raises:
            FileExistsError: in case workspace already exists.
        """
        if workspace_path is None:
            from pytriton.constants import PYTRITON_HOME

            PYTRITON_HOME.mkdir(parents=True, exist_ok=True)
            self._tmp_dir = tempfile.TemporaryDirectory(dir=PYTRITON_HOME, prefix="workspace_")
            self._workspace_path = pathlib.Path(self._tmp_dir.name).resolve()
            LOGGER.debug(f"Workspace path {self._workspace_path}")
        else:
            self._tmp_dir = None
            self._workspace_path = pathlib.Path(workspace_path).resolve()
            LOGGER.debug(f"Workspace path {self._workspace_path}")
            self._workspace_path.mkdir(parents=True)

    @property
    def path(self) -> pathlib.Path:
        """Return path to the workspace.

        Returns:
            Path object with location of workspace catalog
        """
        return self._workspace_path

    def exists(self) -> bool:
        """Verify if workspace catalog exists.

        Returns:
            True if workspace catalog exists. False otherwise.
        """
        return self._workspace_path.exists()

    def is_empty(self) -> bool:
        """Verify if workspace contains any files or folders.

        Returns:
            True if workspace is not empty. False otherwise.
        """
        all_files = list(self.path.rglob("*"))
        if len(all_files) == 0:
            return True
        for p in all_files:
            rel_p = p.relative_to(self.path)
            if rel_p.parts and not rel_p.parts[0].startswith("."):
                return False
        return True

    def clean(self) -> None:
        """Clean workspace removing files and directories created inside including the workspace itself.

        Raises:
            OSError - when workspace after performing cleanup operation is still not empty.
        """
        LOGGER.debug(f"Cleaning workspace dir {self.path}")

        for child in self.path.rglob("*"):
            rel_p = child.relative_to(self.path)
            if len(rel_p.parts) == 0 or rel_p.parts[0].startswith("."):
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        if not self.is_empty():
            raise OSError(f"Could not clean {self.path} workspace")
        if self.path.exists():
            self.path.rmdir()
