# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Triton Inference Server configuration class.

Use to configure the CLI argument for starting the Triton Inference Server process.

    Examples of use:

        config = TritonServerConfig()
        config["log-verbose"] = 1
        config.to_cli_string()
"""
from typing import Any, Dict, List, Optional

from pytriton.exceptions import PyTritonError


class TritonServerConfig:
    """A config class to set arguments to the Triton Inference Server.

    An argument set to None will use the server default.
    """

    # https://github.com/triton-inference-server/server/blob/main/src/command_line_parser.cc
    server_arg_keys = [
        # Server
        "id",
        # Logging
        "log-verbose",
        "log-info",
        "log-warning",
        "log-error",
        "log-format",
        "log-file",
        # Model Repository
        "model-store",
        "model-repository",
        # Exit
        "exit-timeout-secs",
        "exit-on-error",
        # Strictness
        "disable-auto-complete-config",
        "strict-model-config",
        "strict-readiness",
        # http options
        "allow-http",
        "http-address",
        "http-port",
        "reuse-http-port",
        "http-header-forward-pattern",
        "http-thread-count",
        # grpc options
        "allow-grpc",
        "grpc-address",
        "grpc-port",
        "reuse-grpc-port",
        "grpc-header-forward-pattern",
        "grpc-infer-allocation-pool-size",
        "grpc-use-ssl",
        "grpc-use-ssl-mutual",
        "grpc-server-cert",
        "grpc-server-key",
        "grpc-root-cert",
        "grpc-infer-response-compression-level",
        "grpc-keepalive-time",
        "grpc-keepalive-timeout",
        "grpc-keepalive-permit-without-calls",
        "grpc-http2-max-pings-without-data",
        "grpc-http2-min-recv-ping-interval-without-data",
        "grpc-http2-max-ping-strikes",
        "grpc-restricted-protocol",
        # metrics options
        "allow-metrics",
        "allow-gpu-metrics",
        "allow-cpu-metrics",
        "metrics-interval-ms",
        "metrics-port",
        "metrics-address",
        # Model control
        "model-control-mode",
        "repository-poll-secs",
        "load-model",
        # Memory and GPU
        "pinned-memory-pool-byte-size",
        "cuda-memory-pool-byte-size",
        "min-supported-compute-capability",
        "buffer-manager-thread-count",
        # Backend config
        "backend-directory",
        "backend-config",
        "allow-soft-placement",
        "gpu-memory-fraction",
        "tensorflow-version",
        # SageMaker integration
        "allow-sagemaker",
        "sagemaker-port",
        "sagemaker-safe-port-range",
        "sagemaker-thread-count",
        # VertexAI integration
        "allow-vertex-ai",
        "vertex-ai-port",
        "vertex-ai-thread-count",
        "vertex-ai-default-model",
        "metrics-config",
        "trace-config",
        "cache-config",
        "cache-directory",
    ]

    def __init__(self):
        """Construct TritonServerConfig."""
        self._server_args = {}

    @classmethod
    def allowed_keys(cls):
        """Return the list of available server arguments with snake cased options.

        Returns:
            List of str. The keys that can be used to configure tritonserver instance
        """
        snake_cased_keys = [key.replace("-", "_") for key in cls.server_arg_keys]
        return cls.server_arg_keys + snake_cased_keys

    def update_config(self, params: Optional[Dict] = None) -> None:
        """Allows setting values from a params dict.

        Args:
            params: The keys are allowed args to perf_analyzer
        """
        if params:
            for key in params:
                self[key.strip().replace("_", "-")] = params[key]

    def to_cli_string(self) -> str:
        """Utility function to convert a config into a string of arguments to the server with CLI.

        Returns:
            The command consisting of all set arguments to the tritonserver.
            e.g. '--model-repository=/models --log-verbose=True'
        """
        cli_items = []
        for key, val in self._server_args.items():
            if val is None:
                continue
            if isinstance(val, (tuple, list)):
                for sub_val in val:
                    cli_items.append(f"--{key}={sub_val}")
            else:
                cli_items.append(f"--{key}={val}")
        return " ".join(cli_items)

    def to_args_list(self) -> List:
        """Utility function to convert a cli string into a list of arguments.

        The function is taking into account "smart" delimiters. Notice in the example below that only the first equals
        sign is used as split delimiter.

        Returns:
            The list of arguments consisting of all set arguments to the tritonserver.

            Example:
            input cli_string: "--model-control-mode=explicit
                --backend-config=tensorflow,version=2"
            output: ['--model-control-mode', 'explicit',
                '--backend-config', 'tensorflow,version=2']
        """
        args_list = []
        args = self.to_cli_string().split()
        for arg in args:
            args_list += arg.split("=", 1)
        return args_list

    def copy(self) -> "TritonServerConfig":
        """Create copy of config.

        Returns:
            TritonServerConfig object that has the same args as this one
        """
        config_copy = TritonServerConfig()
        config_copy.update_config(params=self._server_args)
        return config_copy

    def server_args(self) -> Dict:
        """Return the dict with defined server arguments.

        Returns:
            Dict where keys are server arguments values are their values
        """
        return self._server_args

    def __getitem__(self, key: str) -> Any:
        """Gets an arguments value in config.

        Args:
            key: The name of the argument to the tritonserver

        Returns:
            The value that the argument is set to in this config
        """
        kebab_cased_key = key.strip().replace("_", "-")
        return self._server_args.get(kebab_cased_key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets an arguments value in config after checking if defined/supported.

        Args:
            key: The name of the argument to the tritonserver
            value: The value to which the argument is being set

        Raises:
            PyTritonError: if key is unsupported or undefined in the config class
        """
        kebab_cased_key = key.strip().replace("_", "-")
        if kebab_cased_key in self.server_arg_keys:
            self._server_args[kebab_cased_key] = value
        else:
            raise PyTritonError(
                f"The argument {key!r} to the Triton Inference " "Server is not supported by the pytriton."
            )

    def __contains__(self, key: str) -> bool:
        """Checks if an argument is defined in the TritonServerConfig.

        Args:
            key: The name of the attribute to check for definition in TritonServerConfig

        Returns:
            True if the argument is defined in the config, False otherwise
        """
        kebab_cased_key = key.strip().replace("_", "-")
        value = self._server_args.get(kebab_cased_key, None)
        return value is not None
