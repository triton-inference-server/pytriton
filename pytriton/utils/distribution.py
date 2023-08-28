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
"""Set of utils to obtain properties of pytriton distribution."""
import logging
import pathlib
import site

LOGGER = logging.getLogger(__name__)


def get_root_module_path() -> pathlib.Path:
    """Obtain path to pytriton module.

    Returns:
        Path to pytriton root module in site or if installed in editable model - local.
    """
    pytriton_module_path = pathlib.Path(__file__).parent.parent
    LOGGER.debug("Obtained pytriton module path: %s", pytriton_module_path)
    return pytriton_module_path


def is_editable_install() -> bool:
    """Checks if pytriton is installed in editable mode.

    Returns:
        True if pytriton is installed in editable mode, False otherwise.
    """
    editable_mode = True
    site_packages = site.getsitepackages() + [site.getusersitepackages()]
    pytriton_module_path = get_root_module_path()
    for site_package in site_packages:
        try:
            pytriton_module_path.relative_to(site_package)
            editable_mode = False
            break
        except ValueError:
            pass
    LOGGER.debug("pytriton is installed in editable mode: %s", editable_mode)
    return editable_mode


def get_libs_path():
    """Obtains path to directory with external libraries required by library.

    Returns:
        Path to directory with external libraries required by library.
    """
    pytriton_module_path = get_root_module_path()
    if is_editable_install():
        libs_path = pytriton_module_path / "tritonserver/external_libs"
    else:
        libs_path = pytriton_module_path.parent / "nvidia_pytriton.libs"
    LOGGER.debug("Obtained nvidia_pytriton.libs path: %s", libs_path)
    return libs_path


def get_stub_path(version: str):
    """Obtains path stub file for provided Python interpreter version.

    Args:
        version: Python interpreter version

    Returns:
        Path to stub file for given Python version
    """
    pytriton_module_path = get_root_module_path()
    stub_path = pytriton_module_path / "tritonserver" / "python_backend_stubs" / version / "triton_python_backend_stub"
    LOGGER.debug("Obtained pytriton stubs path for %s: %s", version, stub_path)
    return stub_path
