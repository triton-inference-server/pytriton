# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Telemetry handling module.

This module contains optional import for Open Telemetry and functions to handle it.
"""

import base64
import importlib.util
import json
import logging
from contextlib import contextmanager
from typing import Dict, Generator, List

# Open Telemetry is not mandatory for PyTriton, but it can be used for tracing
# The import in functions breaks telemetry spans handlign in runtime
try:
    import opentelemetry.baggage  # pytype: disable=import-error
    import opentelemetry.trace  # pytype: disable=import-error
    import opentelemetry.trace.propagation.tracecontext  # pytype: disable=import-error
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # pytype: disable=import-error
    from opentelemetry.sdk.resources import Resource  # pytype: disable=import-error
    from opentelemetry.sdk.trace import TracerProvider  # pytype: disable=import-error
    from opentelemetry.sdk.trace.export import (  # pytype: disable=import-error
        BatchSpanProcessor,  # pytype: disable=import-error
    )

    # from opentelemetry import trace, context
    from opentelemetry.trace import (  # pytype: disable=import-error
        NonRecordingSpan,
        SpanContext,
        Status,
        StatusCode,
        TraceFlags,
    )
    from opentelemetry.trace.propagation.tracecontext import (  # pytype: disable=import-error
        TraceContextTextMapPropagator,
    )

except ImportError:
    pass


LOGGER = logging.getLogger(__name__)


_open_telemetry_tracer = None


def set_telemetry_tracer(tracer):
    """Set tracer for Open Telemetry.

    Sets global tracer used by proxy at inference callable side of communication.

    See trace_config parameter for TritonConfig to set also tracing for Triton.
    Function can only be called onece. Second call raises exception.

    Args:
        tracer: Tracer object for Open Telemetry

    Raises:
        ValueError for second and all further calls
    """
    global _open_telemetry_tracer
    if _open_telemetry_tracer is not None:
        raise ValueError("Telemetry tracer is already set")
    LOGGER.debug(f"Setting telemetry tracer: {tracer}")
    _open_telemetry_tracer = tracer


def get_telemetry_tracer():
    """Return telemetry tracer set by set_telemetry_tracer."""
    global _open_telemetry_tracer
    return _open_telemetry_tracer


def get_span_dict(span):
    """Serialize Open Telemetry span for sending over proxy bus."""
    headers = {}
    with opentelemetry.trace.use_span(span, end_on_exit=False):
        ctx = opentelemetry.baggage.set_baggage("zmq", "baggage")
        opentelemetry.trace.propagation.tracecontext.TraceContextTextMapPropagator().inject(headers, ctx)
    return headers


def start_span_from_remote(span_dict: Dict[str, int], name: str):
    """Create new Open Telemetry span from remote span deserialized from proxy.

    The span ownership goes to caller, which MUST call spand end to register
    event in Open Telemetry server.

    Args:
        span_dict: dictionary with fields trace_id and span_id or None
        name: name of new span started

    Returns:
        Open Telemetry span or None if telemetry is not configured or span_dict is None.
    """
    global _open_telemetry_tracer
    if _open_telemetry_tracer is not None:
        ctx = opentelemetry.trace.propagation.tracecontext.TraceContextTextMapPropagator().extract(span_dict)
        return _open_telemetry_tracer.start_span(name, context=ctx)
    else:
        return None


def start_span_from_span(span, name):
    """Create new Open Telemetry span from existing span.

    The span ownership goes to caller, which MUST call spand end to register
    event in Open Telemetry server.

    Args:
        span: Open Telemetry span
        name: name of new span started

    Returns:
        Open Telemetry span
    """
    span_context = SpanContext(
        trace_id=span.context.trace_id,
        span_id=span.context.span_id,
        is_remote=True,
        trace_flags=TraceFlags(0x01),
    )
    ctx = opentelemetry.trace.set_span_in_context(NonRecordingSpan(span_context))
    tracer = get_telemetry_tracer()
    return tracer.start_span(name, context=ctx)


def parse_trace_config(trace_config_list: List[str]):
    """Parse Triton Open Telemetry config.

    The TritonConfig trace_config can be passed here to obtain Open Telemetry resource and
    URL to connect to server.

    Example of configuration:
    ```
    trace_config=[
        "mode=opentelemetry",
        "opentelemetry,url=<your Open Telemetry API URL>",
        "opentelemetry,resource=service.name=<your service name>",
        "opentelemetry,resource=test.key=test.value",
    ]
    ```
    Elements:
      - List MUST contain mode to indicate opentelemetry support.
      - List MUST contain url to allow opening connecion to Open Telemetry server
      - List SHOULD contain service.name to improve logging
      - List SHOULD contain additional keys like test.key.

    Args:
        trace_config_list: list of configuration variable for Tritonconfig
    """
    if not any("mode=opentelemetry" in config for config in trace_config_list):
        raise ValueError("Only opentelemetry mode is supported")
    url_entry = next((config for config in trace_config_list if "opentelemetry,url=" in config), None)
    if url_entry is None:
        raise ValueError("opentelemetry,url is required")
    url = url_entry.split("opentelemetry,url=")[1]

    resource_attributes = {}
    for config in trace_config_list:
        if config.startswith("opentelemetry,resource="):
            resource_str = config.split("opentelemetry,resource=")[1]
            resource_parts = resource_str.split(",")
            for part in resource_parts:
                key, val = part.split("=")
                resource_attributes[key] = val

    LOGGER.debug(f"OpenTelemetry URL: {url}")
    LOGGER.debug(f"Resource Attributes: {resource_attributes}")

    resource = Resource(attributes=resource_attributes)
    return url, resource


@contextmanager
def traced_span(request, span_name, **kwargs) -> Generator[None, None, None]:
    """Context manager handles opening span for request.

    This context manager opens Open Telemetry span for request. The span is
    automatically closed when context manager exits.

    Example of use in inference callable:
    ```
    def inference_callable(requests):
        responses = []
        for request in requests:
            with traced_span(request, "pass-through-get-data"):
                # Execute compute for single request
    ```

    Args:
        request: Request passed to inference callable
        span_name: Name of span to yield
        **kwargs: Additional arguments passed to Open Telemetry tracer
    """
    global _open_telemetry_tracer
    span = request.span
    if span is not None:
        with opentelemetry.trace.use_span(span, end_on_exit=False, record_exception=False):
            with _open_telemetry_tracer.start_as_current_span(span_name, **kwargs):
                yield
    else:
        yield


def build_proxy_tracer_from_triton_config(trace_config):
    """Build OpenTelemetry tracer from TritonConfig trace_config.

    Args:
        trace_config: list of trace configuration variables

    Returns:
        OpenTelemetry tracer
    """
    raise_if_no_telemetry()
    LOGGER.debug(f"Building OpenTelmetry tracer from config: {trace_config}")
    url, resource = parse_trace_config(trace_config)
    LOGGER.debug(f"Creating OpenTelemetry tracer with URL: {url}")
    opentelemetry.trace.set_tracer_provider(
        TracerProvider(
            resource=resource,
            active_span_processor=BatchSpanProcessor(OTLPSpanExporter(endpoint=url)),
        )
    )

    tracer = opentelemetry.trace.get_tracer(__name__)
    return tracer


def raise_if_no_telemetry():
    """Raise ImportError if OpenTelemetry is not installed."""
    # Import added to trigger error for missing package
    if importlib.util.find_spec("opentelemetry.trace") is None:
        pip = "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        raise ImportError(f"OpenTelemetry is not installed. Please install it using '{pip}'.")


def end_span(span, error=None):
    """End Open Telemetry span and set status if error is provided.

    Args:
        span: Open Telemetry span
        error: error message to set in span status
    """
    if span is not None:
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()


class TracableModel:
    """Model class with tracing support.

    This class is base class for model with tracing support. It provides
    methods to start and end span for each inference call.
    """

    def __init__(self):
        """Initialize TracableModel."""
        self._open_telemetry_tracer = None

    def configure_tracing(self, trace_config):
        """Configure tracing for model.

        This method configures OpenTelemetry tracing for model. The trace_config
        is list of configuration variables passed by TritonConfig.

        Args:
            trace_config: list of trace configuration variables
        """
        try:
            raise_if_no_telemetry()

            trace_config_json = base64.b64decode(trace_config).decode("utf-8")
            trace_config_list = json.loads(trace_config_json)
            LOGGER.debug(f"Configuring tracing with {trace_config_list}")

            url, resource = parse_trace_config(trace_config_list)

            opentelemetry.trace.set_tracer_provider(TracerProvider(resource=resource))
            trace_provider = opentelemetry.trace.get_tracer_provider()
            self._open_telemetry_tracer = trace_provider.get_tracer("pbe")
            trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))

        except Exception as e:
            raise ValueError(f"Failed to configure tracing: {e}") from e

    def start_requests_spans(self, triton_requests):
        """Start spans for requests.

        This method starts spans for each request in triton_requests.

        Args:
            triton_requests: list of Triton requests
        """
        if self._open_telemetry_tracer is not None:
            spans = []
            for triton_request in triton_requests:
                context = triton_request.trace().get_context()
                if context is None:
                    context = "{}"
                ctx = TraceContextTextMapPropagator().extract(carrier=json.loads(context))
                span = self._open_telemetry_tracer.start_span("python_backend_execute", context=ctx)
                spans.append(span)
            return spans
        return None

    def end_requests_spans(self, spans, triton_responses_or_error):
        """End spans for requests.

        This method ends spans for each request in triton_requests.

        Args:
            spans: list of spans for requests
            triton_responses_or_error: list of Triton responses or error
        """
        if self._open_telemetry_tracer is not None:
            status = Status(StatusCode.OK)
            if triton_responses_or_error is not None and isinstance(triton_responses_or_error, Exception):
                status = Status(StatusCode.ERROR, str(triton_responses_or_error))
            for span in spans:
                span.set_status(status)
                span.end()
