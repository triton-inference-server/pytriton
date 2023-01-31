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
import dataclasses
import logging
import os
import pathlib
import threading
import threading as th
import typing
from typing import Callable, Dict, Optional, Sequence, Union

import typing_inspect

from pytriton.client import ModelClient
from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT, TRITON_LOCAL_URL
from pytriton.exceptions import PyTritonValidationError
from pytriton.model_config.tensor import Tensor
from pytriton.models.manager import ModelManager
from pytriton.models.model import Model, ModelConfig, ModelEvent
from pytriton.server.model_repository import TritonModelRepository
from pytriton.server.triton_server import TritonServer
from pytriton.server.triton_server_config import TritonServerConfig
from pytriton.utils.distribution import get_libs_path, get_root_module_path
from pytriton.utils.workspace import Workspace

LOGGER = logging.getLogger(__name__)

TRITONSERVER_DIST_DIR = get_root_module_path() / "tritonserver"
MONITORING_PERIOD_SEC = 10

MODEL_URL = "/v2/models/{model_name}"
MODEL_READY_URL = f"{MODEL_URL}/ready/"
MODEL_CONFIG_URL = f"{MODEL_URL}/config/"
MODEL_INFER_URL = f"{MODEL_URL}/infer/"


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
        http_port: The port for the server to listen on for HTTP requests.
        http_address: The address for the http server to binds to.
        http_thread_count: Number of threads handling HTTP requests.
        allow_grpc: Allow the server to listen for GRPC requests.
        grpc_port: The port for the server to listen on for GRPC requests.
        grpc_address: The address for the grpc server to binds to.
        grpc_infer_allocation_pool_size: The maximum number of inference request/response objects
            that remain allocated for reuse
        grpc_use_ssl: Use SSL authentication for GRPC requests.
        grpc_use_ssl_mutual: Use mutual SSL authentication for GRPC requests.
        grpc_server_cert: Path to file holding PEM-encoded server certificate.
            Ignored unless grpc_use_ssl is true.
        grpc_server_key: Path to file holding PEM-encoded server key.
            Ignored unless grpc_use_ssl is true.
        grpc_root_cert: Path to file holding PEM-encoded root certificate.
            Ignored unless grpc_use_ssl is true.
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
        allow_metrics: Allow the server to provide prometheus metrics.
        allow_gpu_metrics: Allow the server to provide GPU metrics.
        allow_cpu_metrics: Allow the server to provide CPU metrics.
        metrics_interval_ms: Metrics will be collected once every <metrics-interval-ms> milliseconds.
        metrics_port: The port reporting prometheus metrics.
        allow_sagemaker: Allow the server to listen for Sagemaker requests.
        sagemaker_port: The port for the server to listen on for Sagemaker requests.
        sagemaker_safe_port_range: Set the allowed port range for endpoints other than the SageMaker endpoints.
        sagemaker_thread_count: Number of threads handling Sagemaker requests.
        allow_vertex_ai: Allow the server to listen for Vertex AI requests.
        vertex_ai_port: The port for the server to listen on for Vertex AI requests.
        vertex_ai_thread_count: Number of threads handling Vertex AI requests.
        vertex_ai_default_model: The name of the model to use for single-model inference requests.
        trace_file: Set the file where trace output will be saved.
        trace_level: Specify a trace level.
        trace_rate: Set the trace sampling rate.
        trace_count: Set the number of traces to be sampled.
        trace_log_frequency: Set the trace log frequency.
        response_cache_byte_size: The size in bytes to allocate for a request/response cache.
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
    http_port: Optional[int] = None
    http_address: Optional[str] = None
    http_thread_count: Optional[int] = None
    allow_grpc: Optional[bool] = None
    grpc_port: Optional[int] = None
    grpc_address: Optional[str] = None
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
    allow_metrics: Optional[bool] = None
    allow_gpu_metrics: Optional[bool] = None
    allow_cpu_metrics: Optional[bool] = None
    metrics_interval_ms: Optional[int] = None
    metrics_port: Optional[int] = None
    allow_sagemaker: Optional[bool] = None
    sagemaker_port: Optional[int] = None
    sagemaker_safe_port_range: Optional[str] = None
    sagemaker_thread_count: Optional[int] = None
    allow_vertex_ai: Optional[bool] = None
    vertex_ai_port: Optional[int] = None
    vertex_ai_thread_count: Optional[int] = None
    vertex_ai_default_model: Optional[str] = None
    trace_file: Optional[pathlib.Path] = None
    trace_level: Optional[str] = None
    trace_rate: Optional[int] = None
    trace_count: Optional[int] = None
    trace_log_frequency: Optional[int] = None
    response_cache_byte_size: Optional[int] = None
    buffer_manager_thread_count: Optional[int] = None

    def __post_init__(self):
        """Validate configuration for early error handling."""
        if self.allow_http not in [True, None] and self.allow_grpc not in [True, None]:
            raise PyTritonValidationError("The `http` or `grpc` endpoint has to be allowed.")

    def to_dict(self):
        """Map config object to dictionary."""
        return dataclasses.asdict(self)

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
        config = {name[len(prefix) :].lower(): value for name, value in os.environ.items() if name.startswith(prefix)}
        fields: Dict[str, dataclasses.Field] = {field.name: field for field in dataclasses.fields(cls)}
        unknown_config_parameters = {name: value for name, value in config.items() if name not in fields}
        for name, value in unknown_config_parameters.items():
            LOGGER.warning(
                f"Ignoring {name}={value} as could not find matching config field. "
                f"Available fields: {', '.join(map(str, fields))}"
            )

        def _cast_value(_field, _value):
            field_type = _field.type
            is_optional = typing_inspect.is_optional_type(field_type)
            if is_optional:
                field_type = field_type.__args__[0]
            return field_type(_value)

        config_with_casted_values = {
            name: _cast_value(fields[name], value) for name, value in config.items() if name in fields
        }
        return cls(**config_with_casted_values)


class Triton:
    """Triton Inference Server for Python models."""

    def __init__(
        self, *, config: Optional[TritonConfig] = None, workspace: Union[Workspace, str, pathlib.Path, None] = None
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
        """

        def _without_none_values(_d):
            return {name: value for name, value in _d.items() if value is not None}

        default_config_dict = _without_none_values(TritonConfig().to_dict())
        env_config_dict = _without_none_values(TritonConfig.from_env().to_dict())
        explicit_config_dict = _without_none_values(config.to_dict() if config else {})
        config_dict = {**default_config_dict, **env_config_dict, **explicit_config_dict}
        self._config = TritonConfig(**config_dict)
        self._workspace = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

        model_repository = TritonModelRepository(path=self._config.model_repository, workspace=self._workspace)
        self._model_manager = ModelManager(model_repository)

        self._triton_server_config = TritonServerConfig()
        config_data = self._config.to_dict()
        for name, value in config_data.items():
            if name not in TritonServerConfig.allowed_keys() or value is None:
                continue

            self._triton_server_config[name] = value

        self._triton_server_config["model_repository"] = model_repository.path.as_posix()
        self._triton_server_config["backend_directory"] = (TRITONSERVER_DIST_DIR / "backends").as_posix()

        self._triton_server = TritonServer(
            path=(TRITONSERVER_DIST_DIR / "bin/tritonserver").as_posix(),
            libs_path=get_libs_path(),
            config=self._triton_server_config,
        )

        self._cv = th.Condition()
        with self._cv:
            self._stopped = True

    def __enter__(self) -> "Triton":
        """Enter the context.

        Returns:
            A Triton object
        """
        return self

    def __exit__(self, *_) -> None:
        """Exit the context stopping the process and cleaning the workspace.

        Args:
            *_: unused arguments
        """
        self.stop()

    def run(self) -> None:
        """Run Triton Inference Server."""
        if not self._triton_server.is_alive():
            self._model_manager.create_models()
            self._triton_server.register_on_exit(self._on_tritonserver_exit)
            atexit.register(self.stop)

            with self._cv:
                self._triton_server.start()
                self._stopped = False

            self._wait_for_models()

    def stop(self) -> None:
        """Stop Triton Inference Server."""
        with self._cv:
            if self._stopped:
                return
            LOGGER.debug("Stopping Triton Inference server and proxy backends")
            self._triton_server.stop()
            self._model_manager.clean()
            self._workspace.clean()
            self._stopped = True
            self._cv.notify_all()

        LOGGER.debug("Stopped Triton Inference server and proxy backends")
        atexit.unregister(self.stop)

    def serve(self, monitoring_period_sec: int = MONITORING_PERIOD_SEC) -> None:
        """Run Triton Inference Server and lock thread for serving requests/response.

        Args:
            monitoring_period_sec: the timeout of monitoring if Triton and models are available.
                Every monitoring_period_sec seconds main thread wakes up and check if triton server and proxy backend
                are still alive and sleep again. If triton or proxy is not alive - method returns.
        """
        self.run()
        with self._cv:
            while self.is_alive():
                self._cv.wait(timeout=monitoring_period_sec)
            self.stop()

    def is_alive(self) -> bool:
        """Verify is deployed models and server are alive.

        Returns:
            True if server and loaded models are alive, False otherwise.
        """
        if not self._triton_server.is_alive():
            return False

        for model in self._model_manager.models:
            if not model.is_alive():
                return False
        return True

    def bind(
        self,
        model_name: str,
        infer_func: Union[Callable, Sequence[Callable]],
        inputs: Sequence[Tensor],
        outputs: Sequence[Tensor],
        model_version: int = 1,
        config: Optional[ModelConfig] = None,
    ) -> None:
        """Create a model with given name and inference function binding into Triton Inference Server.

        More information about model configuration:
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md

        Args:
            infer_func: Inference function to handle request/response from Triton Inference Server
            (or list of inference functions for multi instance model)
            inputs: Definition of model inputs
            outputs: Definition of model outputs
            model_name: Name under which model is available in Triton Inference Server
            model_version: Version of model
            config: Model configuration for Triton Inference Server deployment
        """
        model = Model(
            model_name=model_name,
            model_version=model_version,
            inference_fn=infer_func,
            inputs=inputs,
            outputs=outputs,
            config=config if config else ModelConfig(),
            workspace=self._workspace,
        )
        model.on_model_event(self._on_model_event)

        self._model_manager.add_model(model)

    def _on_model_event(self, model: Model, event: ModelEvent, context: typing.Optional[typing.Any] = None):
        LOGGER.info(f"Received {event} from {model}; context={context}")

        if event in [ModelEvent.RUNTIME_TERMINATING, ModelEvent.RUNTIME_TERMINATED]:
            threading.Thread(target=self.stop).start()

    def _on_tritonserver_exit(self, *_) -> None:
        """Handle the Triton Inference Server process exit.

        Args:
            _: unused arguments
        """
        LOGGER.debug("Got callback that tritonserver process finished")
        self.stop()

    def _wait_for_models(self) -> None:
        """Log loaded models in console to show the available endpoints."""
        if self._config.allow_http in [True, None]:
            protocol = "http"
            port = self._config.http_port or DEFAULT_HTTP_PORT
        else:
            protocol = "grpc"
            port = self._config.grpc_port or DEFAULT_GRPC_PORT

        server_url = f"{protocol}://{TRITON_LOCAL_URL}:{port}"

        for model in self._model_manager.models:
            with ModelClient(
                url=server_url, model_name=model.model_name, model_version=str(model.model_version)
            ) as client:
                client.wait_for_model(timeout_s=120)
            LOGGER.info(f"Infer function available as model: `{MODEL_URL.format(model_name=model.model_name)}`")
            LOGGER.info(f"  Status:         `GET  {MODEL_READY_URL.format(model_name=model.model_name)}`")
            LOGGER.info(f"  Model config:   `GET  {MODEL_CONFIG_URL.format(model_name=model.model_name)}`")
            LOGGER.info(f"  Inference:      `POST {MODEL_INFER_URL.format(model_name=model.model_name)}`")
