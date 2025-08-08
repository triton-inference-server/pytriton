# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
"""Triton Inference Server class.

The class provide functionality to run Triton Inference Server, load the Python models and serve the requests/response
for models inference.

    Examples of use:
        with Triton() as triton:
            triton.bind(
                model_name="BERT",
                infer_func=_infer_fn,
                inputs=[
                    Tensor(dtype=np.bytes_, shape=(1,)),
                ],
                outputs=[
                    Tensor(dtype=np.float32, shape=(-1,)),
                ],
                config=PythonModelConfig(max_batch_size=16),
            )
            triton.serve()
"""

import atexit
import codecs
import contextlib
import dataclasses
import logging
import os
import pathlib
import re
import secrets
import shutil
import sys
import threading as th
import typing
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import typing_inspect

from pytriton.client import ModelClient
from pytriton.client.utils import TritonUrl, create_client_from_url, wait_for_server_ready
from pytriton.constants import DEFAULT_TRITON_STARTUP_TIMEOUT_S
from pytriton.decorators import TritonContext
from pytriton.exceptions import PyTritonValidationError
from pytriton.model_config.tensor import Tensor
from pytriton.models.manager import ModelManager
from pytriton.models.model import Model, ModelConfig, ModelEvent
from pytriton.proxy.telemetry import build_proxy_tracer_from_triton_config, get_telemetry_tracer, set_telemetry_tracer
from pytriton.server.python_backend_config import PythonBackendConfig
from pytriton.server.triton_server import TritonServer
from pytriton.server.triton_server_config import TritonServerConfig
from pytriton.utils import endpoint_utils
from pytriton.utils.dataclasses import kwonly_dataclass
from pytriton.utils.distribution import get_libs_path, get_root_module_path, get_stub_path
from pytriton.utils.workspace import Workspace

LOGGER = logging.getLogger(__name__)

TRITONSERVER_DIST_DIR = get_root_module_path() / "tritonserver"
MONITORING_PERIOD_S = 10.0
WAIT_FORM_MODEL_TIMEOUT_S = 60.0
INITIAL_BACKEND_SHM_SIZE = 4194304  # 4MB, Python Backend default is 64MB, but is automatically increased
GROWTH_BACKEND_SHM_SIZE = 1048576  # 1MB, Python Backend default is 64MB

MODEL_URL = "/v2/models/{model_name}"
MODEL_READY_URL = f"{MODEL_URL}/ready/"
MODEL_CONFIG_URL = f"{MODEL_URL}/config/"
MODEL_INFER_URL = f"{MODEL_URL}/infer/"

# Valid Triton Server endpoint names for restriction configuration
VALID_TRITON_ENDPOINTS = {
    "health",
    "metadata",
    "inference",
    "shared-memory",
    "model-config",
    "model-repository",
    "statistics",
    "trace",
    "logging",
}

# Default endpoints to protect with token-based access restriction
DEFAULT_PROTECTED_ENDPOINTS = [
    "shared-memory",
    "model-repository",
    "statistics",
    "trace",
    "logging",
]

# Header name for token-based access restriction
TRITON_ACCESS_HEADER = "triton-access" + "-token"


# see https://github.com/triton-inference-server/server/blob/main/src/command_line_parser.cc for more details
@kwonly_dataclass
@dataclasses.dataclass
class TritonConfig:
    """Triton Inference Server configuration class for customization of server execution.

    The arguments are optional. If value is not provided the defaults for Triton Inference Server are used.
    Please, refer to https://github.com/triton-inference-server/server/ for more details.

    Args:
        id: Identifier for this server.
        log_verbose: Set verbose logging level. Zero (0) disables verbose logging and
            values >= 1 enable verbose logging.
        log_file: Set the name of the log output file.
        exit_timeout_secs: Timeout (in seconds) when exiting to wait for in-flight inferences to finish.
        exit_on_error: Exit the inference server if an error occurs during initialization.
        strict_readiness: If true /v2/health/ready endpoint indicates ready if the server is
            responsive and all models are available.
        allow_http: Allow the server to listen for HTTP requests.
        http_address: The address for the http server to bind to. Default is 0.0.0.0.
        http_port: The port for the server to listen on for HTTP requests. Default is 8000.
        http_header_forward_pattern: The regular expression pattern
            that will be used for forwarding HTTP headers as inference request parameters.
        http_thread_count: Number of threads handling HTTP requests.
        http_restricted_api: Specify restricted HTTP API setting.
            The format of this flag is `<APIs>,<key>=<value>`.
            Where `<APIs>` is a comma-separated list of APIs to be restricted.
            `<key>` will be additional header key to be checked when an HTTP request
            is received, and `<value>` is the value expected to be matched.
        allow_grpc: Allow the server to listen for GRPC requests.
        grpc_address: The address for the grpc server to binds to. Default is 0.0.0.0.
        grpc_port: The port for the server to listen on for GRPC requests. Default is 8001.
        grpc_header_forward_pattern: The regular expression pattern that will be used
            for forwarding GRPC headers as inference request parameters.
        grpc_infer_allocation_pool_size: The maximum number of inference request/response objects
            that remain allocated for reuse. As long as the number of in-flight requests doesn't exceed
            this value there will be no allocation/deallocation of request/response objects.
        grpc_use_ssl: Use SSL authentication for GRPC requests. Default is false.
        grpc_use_ssl_mutual: Use mututal SSL authentication for GRPC requests.
            This option will preempt grpc_use_ssl if it is also specified. Default is false.
        grpc_server_cert: File holding PEM-encoded server certificate. Ignored unless grpc_use_ssl is true.
        grpc_server_key: Path to file holding PEM-encoded server key. Ignored unless grpc_use_ssl is true.
        grpc_root_cert: Path to file holding PEM-encoded root certificate. Ignored unless grpc_use_ssl is true.
        grpc_infer_response_compression_level: The compression level to be used while returning the inference
            response to the peer. Allowed values are none, low, medium and high. Default is none.
        grpc_keepalive_time: The period (in milliseconds) after which a keepalive ping is sent on the transport.
        grpc_keepalive_timeout: The period (in milliseconds) the sender of the keepalive ping waits
            for an acknowledgement.
        grpc_keepalive_permit_without_calls: Allows keepalive pings to be sent even if there are no calls in flight
        grpc_http2_max_pings_without_data: The maximum number of pings that can be sent when there is no
            data/header frame to be sent.
        grpc_http2_min_recv_ping_interval_without_data: If there are no data/header frames being sent on the
            transport, this channel argument on the server side controls the minimum time (in milliseconds) that
            gRPC Core would expect between receiving successive pings.
        grpc_http2_max_ping_strikes: Maximum number of bad pings that the server will tolerate before sending
            an HTTP2 GOAWAY frame and closing the transport.
        grpc_restricted_protocol: Specify restricted GRPC protocol setting.
            The format of this flag is `<protocols>,<key>=<value>`.
            Where `<protocol>` is a comma-separated list of protocols to be restricted.
            `<key>` will be additional header key to be checked when a GRPC request
            is received, and `<value>` is the value expected to be matched.
        allow_metrics: Allow the server to provide prometheus metrics.
        allow_gpu_metrics: Allow the server to provide GPU metrics.
        allow_cpu_metrics: Allow the server to provide CPU metrics.
        metrics_interval_ms: Metrics will be collected once every `<metrics-interval-ms>` milliseconds.
        metrics_port: The port reporting prometheus metrics.
        metrics_address: The address for the metrics server to bind to. Default is the same as http_address.
        allow_sagemaker: Allow the server to listen for Sagemaker requests.
        sagemaker_port: The port for the server to listen on for Sagemaker requests.
        sagemaker_safe_port_range: Set the allowed port range for endpoints other than the SageMaker endpoints.
        sagemaker_thread_count: Number of threads handling Sagemaker requests.
        allow_vertex_ai: Allow the server to listen for Vertex AI requests.
        vertex_ai_port: The port for the server to listen on for Vertex AI requests.
        vertex_ai_thread_count: Number of threads handling Vertex AI requests.
        vertex_ai_default_model: The name of the model to use for single-model inference requests.
        metrics_config: Specify a metrics-specific configuration setting.
            The format of this flag is `<setting>=<value>`. It can be specified multiple times
        trace_config: Specify global or trace mode specific configuration setting.
            The format of this flag is `<mode>,<setting>=<value>`.
            Where `<mode>` is either 'triton' or 'opentelemetry'. The default is 'triton'.
            To specify global trace settings (level, rate, count, or mode), the format would be `<setting>=<value>`.
            For 'triton' mode, the server will use Triton's Trace APIs.
            For 'opentelemetry' mode, the server will use OpenTelemetry's APIs to generate,
            collect and export traces for individual inference requests.
            More details, including supported settings can be found at [Triton trace guide](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/trace.md).
        cache_config: Specify a cache-specific configuration setting.
            The format of this flag is `<cache_name>,<setting>=<value>`.
            Where `<cache_name>` is the name of the cache, such as 'local' or 'redis'.
            Example: `local,size=1048576` will configure a 'local' cache implementation
            with a fixed buffer pool of size 1048576 bytes.
        cache_directory: The global directory searched for cache shared libraries. Default is '/opt/tritonserver/caches'.
            This directory is expected to contain a cache implementation as a shared library with the name 'libtritoncache.so'.
        buffer_manager_thread_count: The number of threads used to accelerate copies and other operations
            required to manage input and output tensor contents.
    """

    model_repository: Optional[pathlib.Path] = None
    id: Optional[str] = None
    log_verbose: Optional[int] = None
    log_file: Optional[pathlib.Path] = None
    exit_timeout_secs: Optional[int] = None
    exit_on_error: Optional[bool] = None
    strict_readiness: Optional[bool] = None
    allow_http: Optional[bool] = None
    http_address: Optional[str] = None
    http_port: Optional[int] = None
    http_header_forward_pattern: Optional[str] = None
    http_thread_count: Optional[int] = None
    http_restricted_api: Optional[str] = None
    allow_grpc: Optional[bool] = None
    grpc_address: Optional[str] = None
    grpc_port: Optional[int] = None
    grpc_header_forward_pattern: Optional[str] = None
    grpc_infer_allocation_pool_size: Optional[int] = None
    grpc_use_ssl: Optional[bool] = None
    grpc_use_ssl_mutual: Optional[bool] = None
    grpc_server_cert: Optional[pathlib.Path] = None
    grpc_server_key: Optional[pathlib.Path] = None
    grpc_root_cert: Optional[pathlib.Path] = None
    grpc_infer_response_compression_level: Optional[str] = None
    grpc_keepalive_time: Optional[int] = None
    grpc_keepalive_timeout: Optional[int] = None
    grpc_keepalive_permit_without_calls: Optional[bool] = None
    grpc_http2_max_pings_without_data: Optional[int] = None
    grpc_http2_min_recv_ping_interval_without_data: Optional[int] = None
    grpc_http2_max_ping_strikes: Optional[int] = None
    grpc_restricted_protocol: Optional[str] = None
    allow_metrics: Optional[bool] = None
    allow_gpu_metrics: Optional[bool] = None
    allow_cpu_metrics: Optional[bool] = None
    metrics_interval_ms: Optional[int] = None
    metrics_port: Optional[int] = None
    metrics_address: Optional[str] = None
    allow_sagemaker: Optional[bool] = None
    sagemaker_port: Optional[int] = None
    sagemaker_safe_port_range: Optional[str] = None
    sagemaker_thread_count: Optional[int] = None
    allow_vertex_ai: Optional[bool] = None
    vertex_ai_port: Optional[int] = None
    vertex_ai_thread_count: Optional[int] = None
    vertex_ai_default_model: Optional[str] = None
    metrics_config: Optional[List[str]] = None
    trace_config: Optional[List[str]] = None
    cache_config: Optional[List[str]] = None
    cache_directory: Optional[str] = None
    buffer_manager_thread_count: Optional[int] = None

    def __post_init__(self):
        """Validate configuration for early error handling."""
        if self.allow_http not in [True, None] and self.allow_grpc not in [True, None]:
            raise PyTritonValidationError("The `http` or `grpc` endpoint has to be allowed.")

    def to_dict(self):
        """Map config object to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TritonConfig":
        """Creates a ``TritonConfig`` instance from an input dictionary. Values are converted into correct types.

        Args:
            config: a dictionary with all required fields

        Returns:
            a ``TritonConfig`` instance
        """
        fields: Dict[str, dataclasses.Field] = {field.name: field for field in dataclasses.fields(cls)}
        unknown_config_parameters = {name: value for name, value in config.items() if name not in fields}
        for name, value in unknown_config_parameters.items():
            LOGGER.warning(
                "Ignoring %s=%s as could not find matching config field. Available fields: %s",
                name,
                value,
                ", ".join(map(str, fields)),
            )

        def _cast_value(_field, _value):
            field_type = _field.type
            is_optional = typing_inspect.is_optional_type(field_type)
            if is_optional:
                field_type = field_type.__args__[0]
            if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                return list(_value) if _value is not None else None
            elif isinstance(_value, str) and isinstance(field_type, type) and issubclass(field_type, list):
                return _value.split(",")
            return field_type(_value)

        config_with_casted_values = {
            name: _cast_value(fields[name], value) for name, value in config.items() if name in fields
        }
        return cls(**config_with_casted_values)

    @classmethod
    def from_env(cls) -> "TritonConfig":
        """Creates TritonConfig from environment variables.

        Environment variables should start with `PYTRITON_TRITON_CONFIG_` prefix. For example:

            PYTRITON_TRITON_CONFIG_GRPC_PORT=45436
            PYTRITON_TRITON_CONFIG_LOG_VERBOSE=4

        Typical use:

            triton_config = TritonConfig.from_env()

        Returns:
            TritonConfig class instantiated from environment variables.
        """
        prefix = "PYTRITON_TRITON_CONFIG_"
        config = {}
        list_pattern = re.compile(r"^(.+?)_(\d+)$")

        for name, value in os.environ.items():
            if name.startswith(prefix):
                key = name[len(prefix) :].lower()
                match = list_pattern.match(key)
                if match:
                    list_key, index = match.groups()
                    index = int(index)
                    if list_key not in config:
                        config[list_key] = []
                    if len(config[list_key]) <= index:
                        config[list_key].extend([None] * (index + 1 - len(config[list_key])))
                    config[list_key][index] = value
                else:
                    config[key] = value

        # Remove None values from lists (in case of non-sequential indexes)
        for key in config:
            if isinstance(config[key], list):
                config[key] = [item for item in config[key] if item is not None]

        return cls.from_dict(config)


@dataclasses.dataclass
class TritonLifecyclePolicy:
    """Triton Inference Server lifecycle policy.

    Indicates when Triton server is launched and where the model store is located (locally or remotely managed by
    Triton server).
    """

    launch_triton_on_startup: bool = True
    local_model_store: bool = False


DefaultTritonLifecyclePolicy = TritonLifecyclePolicy()
VertextAILifecyclePolicy = TritonLifecyclePolicy(launch_triton_on_startup=False, local_model_store=True)


@dataclasses.dataclass
class TritonSecurityConfig:
    """Triton Inference Server security configuration.

    Configuration for token-based access restriction to secure model endpoints.
    """

    access_token: Optional[str] = None
    restricted_endpoints: Optional[List[str]] = None

    def __post_init__(self):
        """Validate security configuration."""
        # Handle None vs empty list: None means use defaults, [] means no restrictions
        if self.restricted_endpoints is None:
            self.restricted_endpoints = DEFAULT_PROTECTED_ENDPOINTS.copy()
        self._validate_endpoint_names(self.restricted_endpoints)

        # Only auto-generate token if we have endpoints to protect
        # This allows for tests to check default None state
        if self.access_token is None and self.restricted_endpoints:
            self.access_token = secrets.token_urlsafe(32)

    @staticmethod
    def _validate_endpoint_names(endpoints: List[str]) -> None:
        """Validate that all endpoint names are recognized by Triton Server.

        Args:
            endpoints: List of endpoint names to validate

        Raises:
            PyTritonValidationError: If any endpoint name is invalid
        """
        invalid_endpoints = set(endpoints) - VALID_TRITON_ENDPOINTS
        if invalid_endpoints:
            valid_endpoints_str = ", ".join(sorted(VALID_TRITON_ENDPOINTS))
            invalid_endpoints_str = ", ".join(sorted(invalid_endpoints))
            raise PyTritonValidationError(
                f"Invalid endpoint names: {invalid_endpoints_str}. Valid endpoints are: {valid_endpoints_str}"
            )


class _LogLevelChecker:
    """Check if log level is too verbose."""

    def __init__(self, url: str, access_token: Optional[str] = None) -> None:
        """Initialize LogLevelChecker.

        Args:
            url: Triton Inference Server URL in form of <scheme>://<host>:<port>
            access_token: Access token for the Triton Inference Server

        Raises:
            PyTritonClientInvalidUrlError: if url is invalid
        """
        self._log_settings = None
        self._url = url
        self._headers = None
        if access_token is not None:
            from pytriton.client.auth import create_auth_headers

            # Parse URL to determine protocol
            triton_url = TritonUrl.from_url(url)
            self._headers = create_auth_headers(access_token, triton_url.scheme)

    def check(self, skip_update: bool = False):
        """Check if log level is too verbose.

        Also obtains wait for server is ready + log settings from server if not already obtained.

        Raises:
            PyTritonClientTimeoutError: if timeout is reached
        """
        if self._log_settings is None and not skip_update:
            with contextlib.closing(create_client_from_url(self._url)) as client:
                wait_for_server_ready(client, timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S, headers=self._headers)
                self._log_settings = client.get_log_settings(headers=self._headers)

        if self._log_settings is not None:
            log_settings = self._log_settings
            log_verbose_level = 0
            if hasattr(log_settings, "settings"):  # grpc client
                for key, value in log_settings.settings.items():
                    if key == "log_verbose_level":
                        log_verbose_level = value.uint32_param
                        break
            else:  # http client
                log_settings = {key: str(value) for key, value in log_settings.items()}
                log_verbose_level = int(log_settings.get("log_verbose_level", 0))
            if log_verbose_level > 0:
                LOGGER.warning(
                    "Triton Inference Server is running with enabled verbose logs (log_verbose_level=%d). "
                    "It may affect inference performance.",
                    log_verbose_level,
                )


class TritonBase:
    """Base class for Triton Inference Server."""

    def __init__(
        self,
        url: str,
        workspace: Union[Workspace, str, pathlib.Path, None] = None,
        triton_lifecycle_policy: TritonLifecyclePolicy = DefaultTritonLifecyclePolicy,
        access_token: Optional[str] = None,
    ):
        """Initialize TritonBase.

        Args:
            url: Triton Inference Server URL in form of <scheme>://<host>:<port>
            workspace: Workspace for storing communication sockets and the other temporary files.
            triton_lifecycle_policy:  policy indicating when Triton server is launched and where the model store is located
            (locally or remotely managed by Triton server).
            access_token: Access token for the Triton Inference Server
        """
        self._triton_lifecycle_policy = triton_lifecycle_policy
        self._workspace = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        self._url = url
        _local_model_config_path = (
            self._workspace.model_store_path if triton_lifecycle_policy.local_model_store else None
        )
        self._model_manager = ModelManager(self._url, _local_model_config_path, access_token=access_token)
        self._cv = th.Condition()
        self._triton_context = TritonContext()
        self._access_token = access_token
        self._log_level_checker = _LogLevelChecker(self._url, access_token=access_token)
        self._headers = None
        if access_token is not None:
            from pytriton.client.auth import create_auth_headers

            # Parse URL to determine protocol
            triton_url = TritonUrl.from_url(self._url)
            self._headers = create_auth_headers(access_token, triton_url.scheme)

        with self._cv:
            self._stopped = True
            self._connected = False

        atexit.register(self.stop)

    def bind(
        self,
        model_name: str,
        infer_func: Union[Callable, Sequence[Callable]],
        inputs: Sequence[Tensor],
        outputs: Sequence[Tensor],
        model_version: int = 1,
        config: Optional[ModelConfig] = None,
        strict: bool = False,
        trace_config: Optional[List[str]] = None,
    ) -> None:
        """Create a model with given name and inference callable binding into Triton Inference Server.

        More information about model configuration:
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md

        Args:
            infer_func: Inference callable to handle request/response from Triton Inference Server
            (or list of inference callable for multi instance model)
            inputs: Definition of model inputs
            outputs: Definition of model outputs
            model_name: Name under which model is available in Triton Inference Server. It can only contain
            alphanumeric characters, dots, underscores and dashes.
            model_version: Version of model
            config: Model configuration for Triton Inference Server deployment
            strict: Enable strict validation between model config outputs and inference function result
            trace_config: List of trace config parameters
        """
        self._validate_model_name(model_name)
        model_kwargs = {}
        if trace_config is None:
            triton_config = getattr(self, "_config", None)
            if triton_config is not None:
                trace_config = getattr(triton_config, "trace_config", None)
                if trace_config is not None:
                    LOGGER.info("Using trace config from TritonConfig: %s", trace_config)
                    model_kwargs["trace_config"] = trace_config
        else:
            model_kwargs["trace_config"] = trace_config
        telemetry_tracer = get_telemetry_tracer()

        # Automatically set telemetry tracer if not set at the proxy side
        if telemetry_tracer is None and trace_config is not None:
            LOGGER.info("Setting telemetry tracer from TritonConfig")
            telemetry_tracer = build_proxy_tracer_from_triton_config(trace_config)
            set_telemetry_tracer(telemetry_tracer)

        model = Model(
            model_name=model_name,
            model_version=model_version,
            inference_fn=infer_func,
            inputs=inputs,
            outputs=outputs,
            config=config if config else ModelConfig(),
            workspace=self._workspace,
            triton_context=self._triton_context,
            strict=strict,
            **model_kwargs,
        )
        model.on_model_event(self._on_model_event)

        self._model_manager.add_model(model, self.is_connected())

    def connect(self) -> None:
        """Connect to Triton Inference Server.

        Raises:
            TimeoutError: if Triton Inference Server is not ready after timeout
        """
        with self._cv:
            if self._connected:
                LOGGER.debug("Triton Inference already connected.")
                return

            self._wait_for_server()
            if self._triton_lifecycle_policy.local_model_store:
                self._model_manager.setup_models()
            else:
                self._model_manager.load_models()

            self._wait_for_models()
            self._connected = True

    def serve(self, monitoring_period_s: float = MONITORING_PERIOD_S) -> None:
        """Run Triton Inference Server and lock thread for serving requests/response.

        Args:
            monitoring_period_s: the timeout of monitoring if Triton and models are available.
                Every monitoring_period_s seconds main thread wakes up and check if triton server and proxy backend
                are still alive and sleep again. If triton or proxy is not alive - method returns.
        """
        self.connect()
        with self._cv:
            try:
                while self.is_alive():
                    self._cv.wait(timeout=monitoring_period_s)
            except KeyboardInterrupt:
                LOGGER.info("SIGINT received, exiting.")
            self.stop()

    def stop(self) -> bool:
        """Stop Triton Inference Server and clean workspace."""
        with self._cv:
            if self._stopped:
                LOGGER.debug("Triton Inference already stopped.")
                return False
            self._stopped = True
            self._connected = False
            atexit.unregister(self.stop)
        self._pre_stop_impl()
        self._model_manager.clean()
        self._workspace.clean()

        with self._cv:
            self._cv.notify_all()
        LOGGER.debug("Stopped Triton Inference server and proxy backends")
        self._log_level_checker.check(skip_update=True)

        return True

    def is_alive(self) -> bool:
        """Check if Triton Inference Server is alive."""
        if not self._is_alive_impl():
            return False

        for model in self._model_manager.models:
            if not model.is_alive():
                return False
        return True

    def is_connected(self) -> bool:
        """Check if Triton Inference Server is connected."""
        with self._cv:
            return self._connected

    def __enter__(self):
        """Connects to Triton server on __enter__.

        Returns:
            A Triton object
        """
        if self._triton_lifecycle_policy.launch_triton_on_startup:
            self.connect()
        return self

    def __exit__(self, *_) -> None:
        """Exit the context stopping the process and cleaning the workspace.

        Args:
            *_: unused arguments
        """
        self.stop()

    def _is_alive_impl(self) -> bool:
        return True

    def _pre_stop_impl(self):
        pass

    def _post_stop_impl(self):
        pass

    def _wait_for_server(self) -> None:
        """Wait for Triton Inference Server to be ready."""
        self._log_level_checker.check()
        try:
            with contextlib.closing(create_client_from_url(self._url)) as client:
                wait_for_server_ready(client, timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S, headers=self._headers)
        except TimeoutError as e:
            LOGGER.warning(
                "Could not verify locally if Triton Inference Server is ready using %s. "
                "Please, check the server logs for details.",
                self._url,
            )
            raise TimeoutError("Triton Inference Server is not ready after timeout.") from e

    def _wait_for_models(self) -> None:
        """Log loaded models in console to show the available endpoints."""
        self._log_level_checker.check()

        try:
            for model in self._model_manager.models:
                with ModelClient(
                    url=self._url,
                    model_name=model.model_name,
                    model_version=str(model.model_version),
                    access_token=self._access_token,
                ) as client:
                    # This waits for only tritonserver and lightweight proxy backend to be ready
                    # timeout should be short as model is loaded before execution of Triton.start() method
                    client.wait_for_model(timeout_s=WAIT_FORM_MODEL_TIMEOUT_S)
        except TimeoutError:
            LOGGER.warning(
                "Could not verify locally if models are ready using %s. Please, check the server logs for details.",
                self._url,
            )

        for model in self._model_manager.models:
            LOGGER.info("Infer function available as model: `%s`", MODEL_URL.format(model_name=model.model_name))
            LOGGER.info("  Status:         `GET  %s`", MODEL_READY_URL.format(model_name=model.model_name))
            LOGGER.info("  Model config:   `GET  %s`", MODEL_CONFIG_URL.format(model_name=model.model_name))
            LOGGER.info("  Inference:      `POST %s`", MODEL_INFER_URL.format(model_name=model.model_name))

        LOGGER.info(
            "Read more about configuring and serving models in "
            "documentation: https://triton-inference-server.github.io/pytriton."
        )
        LOGGER.info("(Press CTRL+C or use the command `kill -SIGINT %d` to send a SIGINT signal and quit)", os.getpid())

    def _on_model_event(self, model: Model, event: ModelEvent, context: typing.Optional[typing.Any] = None):
        LOGGER.info("Received %s from %s; context=%s", event, model, context)
        if event in [ModelEvent.RUNTIME_TERMINATING, ModelEvent.RUNTIME_TERMINATED]:
            th.Thread(target=self.stop).start()

    @classmethod
    def _validate_model_name(cls, model_name: str) -> None:
        """Validate model name.

        Args:
            model_name: Model name
        """
        if not model_name:
            raise PyTritonValidationError("Model name cannot be empty")

        if not re.match(r"^[a-zA-Z0-9._-]+$", model_name):
            raise PyTritonValidationError(
                "Model name can only contain alphanumeric characters, dots, underscores and dashes"
            )


class Triton(TritonBase):
    """Triton Inference Server for Python models."""

    def __init__(
        self,
        *,
        config: Optional[TritonConfig] = None,
        workspace: Union[Workspace, str, pathlib.Path, None] = None,
        triton_lifecycle_policy: Optional[TritonLifecyclePolicy] = None,
        security_config: Optional[TritonSecurityConfig] = None,
    ):
        """Initialize Triton Inference Server context for starting server and loading models.

        Args:
            config: TritonConfig object with optional customizations for Triton Inference Server.
                Configuration can be passed also through environment variables.
                See [TritonConfig.from_env()][pytriton.triton.TritonConfig.from_env] class method for details.

                Order of precedence:

                  - config defined through `config` parameter of init method.
                  - config defined in environment variables
                  - default TritonConfig values
            workspace: workspace or path where the Triton Model Store and files used by pytriton will be created.
                If workspace is `None` random workspace will be created.
                Workspace will be deleted in [Triton.stop()][pytriton.triton.Triton.stop].
            triton_lifecycle_policy:  policy indicating when Triton server is launched and where the model store is located
                (locally or remotely managed by Triton server). If triton_lifecycle_policy is None,
                DefaultTritonLifecyclePolicy is used by default (Triton server is launched on startup and model store is not local).
                Only if triton_lifecycle_policy is None and config.allow_vertex_ai is True, VertextAILifecyclePolicy is used instead.
            security_config: TritonSecurityConfig object with security settings for token-based access restriction.
                If not provided, DefaultTritonSecurityConfig is used, which auto-generates a secure token and protects
                the default endpoints: shared-memory, model-repository, statistics, trace, logging.
        """
        _triton_lifecycle_policy = (
            VertextAILifecyclePolicy
            if triton_lifecycle_policy is None and config is not None and config.allow_vertex_ai
            else triton_lifecycle_policy
        ) or DefaultTritonLifecyclePolicy

        def _without_none_values(_d):
            return {name: value for name, value in _d.items() if value is not None}

        default_config_dict = _without_none_values(TritonConfig().to_dict())
        env_config_dict = _without_none_values(TritonConfig.from_env().to_dict())
        explicit_config_dict = _without_none_values(config.to_dict() if config else {})
        config_dict = {**default_config_dict, **env_config_dict, **explicit_config_dict}
        self._config = TritonConfig(**config_dict)
        self._security_config = security_config or TritonSecurityConfig()

        workspace_instance = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        self._prepare_triton_config(workspace_instance)
        endpoint_protocol = "http" if self._config.allow_http in [True, None] else "grpc"
        super().__init__(
            url=endpoint_utils.get_endpoint(self._triton_server_config, endpoint_protocol),
            workspace=workspace_instance,
            triton_lifecycle_policy=_triton_lifecycle_policy,
            access_token=self._security_config.access_token,
        )
        self._triton_server = None

    def __enter__(self) -> "Triton":
        """Entering the context launches the triton server.

        Returns:
            A Triton object
        """
        if self._triton_lifecycle_policy.launch_triton_on_startup:
            self._run_server()
        super().__enter__()
        return self

    def run(self) -> None:
        """Run Triton Inference Server."""
        self._run_server()
        self.connect()

    def serve(self, monitoring_period_s: float = MONITORING_PERIOD_S) -> None:
        """Run Triton Inference Server and lock thread for serving requests/response.

        Args:
            monitoring_period_s: the timeout of monitoring if Triton and models are available.
                Every monitoring_period_s seconds main thread wakes up and check if triton server and proxy backend
                are still alive and sleep again. If triton or proxy is not alive - method returns.
        """
        self._run_server()
        super().serve(monitoring_period_s=monitoring_period_s)

    def _initialize_server(self) -> None:
        """Initialize Triton Inference Server before binary execution."""
        self._triton_inference_server_path = self._prepare_triton_inference_server()
        self._triton_server = TritonServer(
            path=(self._triton_inference_server_path / "bin" / "tritonserver").as_posix(),
            libs_path=get_libs_path(),
            config=self._triton_server_config,
        )

        url = (
            self._triton_server.get_endpoint("http")
            if (self._config.allow_http is None or self._config.allow_http)
            else self._triton_server.get_endpoint("grpc")
        )
        self._log_level_checker = _LogLevelChecker(url, access_token=self._security_config.access_token)

    def _prepare_triton_config(self, workspace: Workspace) -> None:
        self._triton_server_config = TritonServerConfig()
        config_data = self._config.to_dict()
        self._python_backend_config = PythonBackendConfig()
        python_backend_config_data = {
            "shm-region-prefix-name": self._shm_prefix(),
            "shm-default-byte-size": INITIAL_BACKEND_SHM_SIZE,
            "shm-growth-byte-size": GROWTH_BACKEND_SHM_SIZE,
        }
        for name, value in python_backend_config_data.items():
            if name not in PythonBackendConfig.allowed_keys() or value is None:
                continue

            if isinstance(value, pathlib.Path):
                value = value.as_posix()
            self._python_backend_config[name] = value
        for name, value in config_data.items():
            if name not in TritonServerConfig.allowed_keys() or value is None:
                continue

            if isinstance(value, pathlib.Path):
                value = value.as_posix()
            self._triton_server_config[name] = value

        self._triton_server_config["model_control_mode"] = "explicit"
        self._triton_server_config["load-model"] = "*"
        self._triton_server_config["backend_config"] = self._python_backend_config.to_list_args()
        if "model_repository" not in self._triton_server_config:
            self._triton_server_config["model_repository"] = workspace.model_store_path.as_posix()

        # Configure token-based access restrictions
        # Only apply restrictions if we have a token AND non-empty endpoint list
        # Empty list [] means no restrictions (unrestricted behavior)
        if self._security_config.access_token and self._security_config.restricted_endpoints:
            endpoints_str = ",".join(self._security_config.restricted_endpoints)
            restriction_value = f"{endpoints_str}:{TRITON_ACCESS_HEADER}={self._security_config.access_token}"

            # Set both HTTP and gRPC restrictions with the same configuration
            self._triton_server_config["grpc-restricted-protocol"] = restriction_value
            self._triton_server_config["http-restricted-api"] = restriction_value

    def _prepare_triton_inference_server(self) -> pathlib.Path:
        """Prepare binaries and libraries of Triton Inference Server for execution.

        Return:
            Path where Triton binaries are ready for execution
        """
        triton_inference_server_path = self._workspace.path / "tritonserver"

        LOGGER.debug("Preparing Triton Inference Server binaries and libs for execution.")
        shutil.copytree(
            TRITONSERVER_DIST_DIR,
            triton_inference_server_path,
            ignore=shutil.ignore_patterns("python_backend_stubs", "triton_python_backend_stub"),
        )
        LOGGER.debug("Triton Inference Server binaries copied to %s without stubs.", triton_inference_server_path)

        major = sys.version_info[0]
        minor = sys.version_info[1]
        version = f"{major}.{minor}"

        src_stub_path = get_stub_path(version)
        dst_stub_path = triton_inference_server_path / "backends" / "python" / "triton_python_backend_stub"

        LOGGER.debug("Copying stub for version %s from %s to %s", version, src_stub_path, dst_stub_path)
        shutil.copy(src_stub_path, dst_stub_path)

        LOGGER.debug("Triton Inference Server binaries ready in %s", triton_inference_server_path)

        self._triton_server_config["backend_directory"] = (triton_inference_server_path / "backends").as_posix()
        if "cache_directory" not in self._triton_server_config:
            self._triton_server_config["cache_directory"] = (triton_inference_server_path / "caches").as_posix()
        return triton_inference_server_path

    def _shm_prefix(self) -> str:
        """Generate unique prefix for shm memory.

        Returns:
            String with prefix
        """
        hash = codecs.encode(os.urandom(4), "hex").decode()
        pid = os.getpid()
        return f"pytrtion{pid}-{hash}"

    def _run_server(self):
        """Run Triton Inference Server."""
        if self._triton_server is None:
            self._initialize_server()
        if not self._triton_server.is_alive():
            with self._cv:
                self._stopped = False
            LOGGER.debug("Starting Triton Inference")
            self._triton_server.register_on_exit(self._on_tritonserver_exit)
            self._triton_server.start()

    def _is_alive_impl(self) -> bool:
        """Verify is deployed models and server are alive.

        Returns:
            True if server and loaded models are alive, False otherwise.
        """
        if not self._triton_server:
            return False

        return self._triton_server.is_alive()

    def _pre_stop_impl(self):
        self._triton_server.unregister_on_exit(self._on_tritonserver_exit)
        if self._triton_server is not None:
            self._triton_server.stop()

    def _on_tritonserver_exit(self, *_) -> None:
        """Handle the Triton Inference Server process exit.

        Args:
            _: unused arguments
        """
        LOGGER.debug("Got callback that tritonserver process finished")
        self.stop()

    def get_access_token(self) -> str:
        """Get the current access token for endpoint restrictions.

        Returns:
            The access token string (explicit or generated)
        """
        return self._security_config.access_token


class RemoteTriton(TritonBase):
    """RemoteTriton connects to Triton Inference Server running on remote host."""

    def __init__(
        self, url: str, workspace: Union[Workspace, str, pathlib.Path, None] = None, access_token: Optional[str] = None
    ):
        """Initialize RemoteTriton.

        Args:
            url: Triton Inference Server URL in form of <scheme>://<host>:<port>
                If scheme is not provided, http is used as default.
                If port is not provided, 8000 is used as default for http and 8001 for grpc.
            workspace: path to be created where the files used by pytriton will be stored
                (e.g. socket files for communication).
                If workspace is `None` temporary workspace will be created.
                Workspace should be created in shared filesystem space between RemoteTriton
                and Triton Inference Server to allow access to socket files
                (if you use containers, folder must be shared between containers).
            access_token: Access token for the Triton Inference Server
        """
        super().__init__(
            url=TritonUrl.from_url(url).with_scheme,
            workspace=workspace,
            triton_lifecycle_policy=TritonLifecyclePolicy(launch_triton_on_startup=True, local_model_store=False),
            access_token=access_token,
        )

        with self._cv:
            self._stopped = False

    def __enter__(self) -> "RemoteTriton":
        """Entering the context connects to remote Triton server.

        Returns:
            A RemoteTriton object
        """
        super().__enter__()
        return self
