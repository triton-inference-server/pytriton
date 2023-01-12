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
import pathlib
import site


def _get_site_packages() -> pathlib.Path:
    """Obtains path to current interpreter site-packages directory where pytriton is installed.

    Returns:
        Path to current interpreter site directory where pytriton is installed.
    """

    def _has_pytriton_installed(site_package_path: str):
        site_package_path = pathlib.Path(site_package_path)
        return bool(list(site_package_path.glob("*pytriton*")))

    site_packages_with_pytriton = [pathlib.Path(p) for p in site.getsitepackages() if _has_pytriton_installed(p)]
    assert site_packages_with_pytriton, f"Could not find pytriton package installed in {site.getsitepackages()}"
    return site_packages_with_pytriton[0]


def get_libs_path():
    """Obtains path to directory with external libraries required by library.

    Returns:
        Path to directory with external libraries required by library.
    """
    if is_editable_install():
        return get_root_module_path() / "tritonserver/external_libs"
    else:
        return _get_site_packages() / "pytriton.libs"


def get_root_module_path() -> pathlib.Path:
    """Obtain path to pytriton module.

    Returns:
        Path to pytriton root module in site or if installed in editable model - local.
    """
    return pathlib.Path(__file__).parent.parent


def is_editable_install() -> bool:
    """Checks if pytriton is installed in editable mode.

    Returns:
        True if pytriton is installed in editable model, False otherwise.
    """
    import pytriton

    pytriton_module_path = pathlib.Path(list(pytriton.__path__)[0]).parent
    try:
        pytriton_module_path.relative_to(_get_site_packages())
        return False
    except ValueError:
        return True
