# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Python Backend configuration class.

Use to configure the CLI argument for Python Backend passed on Triton Inference Server process start.

    Examples of use:

        config = PythonBackendConfig()
        config["shm-default-byte-size"] = 33554432
        config.to_list_args() # ["python,shm-default-byte-size=33554432"]
"""
from typing import Any, Dict, List, Optional, Union

from pytriton.exceptions import PyTritonError


class PythonBackendConfig:
    """A config class to set arguments to the Triton Inference Server.

    An argument set to None will use the server default.
    """

    backend_arg_keys = [
        "shm-region-prefix-name",
        "shm-default-byte-size",
        "shm-growth-byte-size",
    ]

    def __init__(self):
        """Construct PythonBackendConfig."""
        self._backend_args = {}

    @classmethod
    def allowed_keys(cls):
        """Return the list of available server arguments with snake cased options.

        Returns:
            List of str. The keys that can be used to configure Python Backend instance
        """
        snake_cased_keys = [key.replace("-", "_") for key in cls.backend_arg_keys]
        return cls.backend_arg_keys + snake_cased_keys

    @classmethod
    def backend_keys(cls):
        """Return the list of available server arguments with snake cased options.

        Returns:
            List of str. The keys that can be used to configure Python Backend instance
        """
        snake_cased_keys = [key.replace("-", "_") for key in cls.backend_arg_keys]
        return cls.backend_arg_keys + snake_cased_keys

    def update_config(self, params: Optional[Dict] = None) -> None:
        """Allows setting values from a params dict.

        Args:
            params: The keys are allowed args to perf_analyzer
        """
        if params:
            for key in params:
                self[key.strip().replace("_", "-")] = params[key]

    def to_list_args(self) -> List[str]:
        """Utility function to convert a config into a list of arguments to the server with CLI.

        Returns:
            The command consisting of all set arguments to the Python Backend.
            e.g. ['python,shm-default-byte-size=33554432']
        """
        cli_items = []
        for key, val in self._backend_args.items():
            if val is None:
                continue
            cli_items.append(f"python,{key}={val}")

        return cli_items

    def copy(self) -> "PythonBackendConfig":
        """Create copy of config.

        Returns:
            PythonBackendConfig object that has the same args as this one
        """
        config_copy = PythonBackendConfig()
        config_copy.update_config(params=self._backend_args)
        return config_copy

    def backend_args(self) -> Dict:
        """Return the dict with defined server arguments.

        Returns:
            Dict where keys are server arguments values are their values
        """
        return self._backend_args

    def __getitem__(self, key: str) -> Any:
        """Gets an arguments value in config.

        Args:
            key: The name of the argument to the Python Backend

        Returns:
            The value that the argument is set to in this config
        """
        kebab_cased_key = key.strip().replace("_", "-")
        return self._backend_args.get(kebab_cased_key, None)

    def __setitem__(self, key: str, value: Union[str, int]) -> None:
        """Sets an arguments value in config after checking if defined/supported.

        Args:
            key: The name of the argument to the Python Backend
            value: The value to which the argument is being set

        Raises:
            PyTritonError: if key is unsupported or undefined in the config class
        """
        assert isinstance(value, int) or isinstance(value, str)

        kebab_cased_key = key.strip().replace("_", "-")
        if kebab_cased_key in self.backend_arg_keys:
            self._backend_args[kebab_cased_key] = value
        else:
            raise PyTritonError(f"The argument {key!r} to the Python Backend is not supported by the pytriton.")

    def __contains__(self, key: str) -> bool:
        """Checks if an argument is defined in the PythonBackendConfig.

        Args:
            key: The name of the attribute to check for definition in PythonBackendConfig

        Returns:
            True if the argument is defined in the config, False otherwise
        """
        kebab_cased_key = key.strip().replace("_", "-")
        value = self._backend_args.get(kebab_cased_key, None)
        return value is not None
