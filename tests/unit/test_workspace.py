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
import multiprocessing
import os.path
import pathlib
import shutil
import tempfile

import pytest


def test_workspace_exist_and_empty_when_created():
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")

        assert workspace._workspace_path.exists()
        assert workspace.model_store_path.exists()
        assert workspace.exists()
        assert not workspace.is_empty()
        assert len(os.listdir(workspace._workspace_path)) == 1
        assert workspace.path == workspace._workspace_path


def test_workspace_not_exist_and_empty_when_removed():
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")

        shutil.rmtree(workspace._workspace_path.as_posix())
        assert not workspace.exists()
        assert workspace.is_empty()


def test_workspace_initializer_raises_error_when_workspace_directory_already_exists():
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")

        # Exception should be raised at second creation in the same path
        with pytest.raises(FileExistsError):
            _ = Workspace(workspace.path)


def test_workspace_clean():
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")

        open(workspace.path / "file.txt", "w").close()
        (workspace.path / "dir").mkdir()
        assert not workspace.is_empty()

        workspace.clean()
        assert workspace.is_empty()
        assert not workspace.exists()

        # No exception should be raised at second clean
        workspace.clean()
        assert not workspace.exists()


def test_tmp_workspace_exist_when_created():
    import pytriton.constants as constants
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        constants.PYTRITON_HOME = pathlib.Path(tempdir)
        workspace = Workspace()
        assert workspace.exists()


def test_tmp_workspace_not_exist_when_deleted():
    import pytriton.constants as constants
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        constants.PYTRITON_HOME = pathlib.Path(tempdir)
        workspace = Workspace()
        assert workspace.exists()
        p = workspace.path
        del workspace
        assert not p.exists()


def test_tmp_workspace_not_exist_when_cleaned():
    import pytriton.constants as constants
    from pytriton.utils.workspace import Workspace

    with tempfile.TemporaryDirectory() as tempdir:
        constants.PYTRITON_HOME = pathlib.Path(tempdir)
        workspace = Workspace()
        assert workspace.exists()
        workspace.clean()
        assert not workspace.exists()


def _check(pytriton_home=None):
    if pytriton_home is not None:
        os.environ["PYTRITON_HOME"] = pytriton_home
    else:
        pytriton_home = pathlib.Path.home() / ".cache" / "pytriton"

    from pytriton.constants import PYTRITON_HOME

    assert PYTRITON_HOME == pathlib.Path(pytriton_home), f"{PYTRITON_HOME} != {pytriton_home}"

    from pytriton.utils.workspace import Workspace

    workspace = Workspace()
    # below line will raise ValueError if workspace is not in pytriton_home
    workspace.path.relative_to(pytriton_home)


def test_use_env_variable_to_set_pytriton_home():
    ctx = multiprocessing.get_context("spawn")

    process = ctx.Process(target=_check)  # no pytriton_home environment
    process.start()
    process.join()
    assert process.exitcode == 0

    with tempfile.TemporaryDirectory() as tempdir:
        process = ctx.Process(target=_check, args=(tempdir,))
        process.start()
        process.join()
        assert process.exitcode == 0
