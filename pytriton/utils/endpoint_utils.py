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
"""Endpoint url forming utilities module."""
import re
from typing import Literal

from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT, DEFAULT_METRICS_PORT, TRITON_LOCAL_IP
from pytriton.server.triton_server_config import TritonServerConfig


def get_endpoint(server_config: TritonServerConfig, endpoint: Literal["http", "grpc", "metrics"]) -> str:
    """Get endpoint url.

    Args:
        server_config: TritonServerConfig object
        endpoint: endpoint name

    Returns:
        endpoint url in form of {protocol}://{host}:{port}
    """
    protocols = {"http": "http", "grpc": "grpc", "metrics": "http"}

    def _obtain_address(key_names):
        for key_name in key_names:
            address = server_config[key_name]
            if address and not re.match(r"^0+.0+.0+.0+$", address):
                break
        else:
            address = TRITON_LOCAL_IP

        return address

    addresses = {
        "http": _obtain_address(["http-address"]),
        "grpc": _obtain_address(["grpc-address"]),
        "metrics": _obtain_address(["metrics-address", "http-address"]),
    }
    ports = {
        "http": server_config["http-port"] or DEFAULT_HTTP_PORT,
        "grpc": server_config["grpc-port"] or DEFAULT_GRPC_PORT,
        "metrics": server_config["metrics-port"] or DEFAULT_METRICS_PORT,
    }

    return f"{protocols[endpoint]}://{addresses[endpoint]}:{ports[endpoint]}"
