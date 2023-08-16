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
import pathlib
import site

from pytriton.utils.distribution import get_libs_path, get_root_module_path, is_editable_install


def test_is_editable_install(mocker):
    """Test if pytriton is installed in editable mode."""
    mocker.patch("pytriton.utils.distribution.get_root_module_path", return_value=pathlib.Path("/home/user/pytriton"))
    assert is_editable_install()

    mocker.patch(
        "pytriton.utils.distribution.get_root_module_path",
        return_value=pathlib.Path(f"{site.getsitepackages()[0]}/pytriton"),
    )
    assert not is_editable_install()

    mocker.patch(
        "pytriton.utils.distribution.get_root_module_path",
        return_value=pathlib.Path(f"{site.getusersitepackages()}/pytriton"),
    )
    assert not is_editable_install()


def test_get_root_module_path(mocker):
    """Test obtaining path to pytriton module."""
    expected_value = pathlib.Path("/home/user/pytriton")

    # mock obtaining path of pytriton/utils/distribution.py file
    mocker.patch(
        "pytriton.utils.distribution.pathlib.Path",
        return_value=pathlib.Path("/home/user/pytriton/utils/distribution.py"),
    )
    assert get_root_module_path() == expected_value


def test_get_libs_path(mocker):
    """Test obtaining path to directory with external libraries required by Triton."""
    mocker.patch("pytriton.utils.distribution.get_root_module_path", return_value=pathlib.Path("/home/user/pytriton"))
    assert get_libs_path() == pathlib.Path("/home/user/pytriton/tritonserver/external_libs")

    mocker.patch(
        "pytriton.utils.distribution.get_root_module_path",
        return_value=pathlib.Path(f"{site.getsitepackages()[0]}/pytriton"),
    )
    assert get_libs_path() == pathlib.Path(f"{site.getsitepackages()[0]}/nvidia_pytriton.libs")

    mocker.patch(
        "pytriton.utils.distribution.get_root_module_path",
        return_value=pathlib.Path(f"{site.getusersitepackages()}/pytriton"),
    )
    assert get_libs_path() == pathlib.Path(f"{site.getusersitepackages()}/nvidia_pytriton.libs")
