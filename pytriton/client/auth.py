# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""Client authentication utilities for PyTriton.

This module provides client-specific authentication functionality including:
- Protocol-aware header generation for HTTP and gRPC clients
- Token extraction from authentication headers
"""

from typing import Dict, Optional

from pytriton.triton import TRITON_ACCESS_HEADER


def extract_access_token_from_headers(headers: Optional[Dict[str, str]]) -> Optional[str]:
    """Extract access token from authentication headers.

    This function tries to extract an access token from headers that may contain
    either HTTP or gRPC format authentication headers:

    - HTTP format: {"triton-access-token": "token-value"}
    - gRPC format: {"triton-grpc-protocol-triton-access-token": "token-value"}

    Args:
        headers: Dictionary containing headers to search for access token

    Returns:
        The access token if found, None otherwise

    Raises:
        ValueError: If HTTP and gRPC format tokens are present and differ

    Examples:
        >>> headers = {"triton-access-token": "my-token"}
        >>> extract_access_token_from_headers(headers)
        "my-token"

        >>> headers = {"triton-grpc-protocol-triton-access-token": "my-token"}
        >>> extract_access_token_from_headers(headers)
        "my-token"

        >>> headers = {"other-header": "value"}
        >>> extract_access_token_from_headers(headers)
        None
    """
    if not headers:
        return None

    http_token = headers.get(TRITON_ACCESS_HEADER, None)
    grpc_header_name = f"triton-grpc-protocol-{TRITON_ACCESS_HEADER}"
    grpc_token = headers.get(grpc_header_name, None)

    # If both are present, they must match
    if http_token is not None and grpc_token is not None and http_token != grpc_token:
        raise ValueError(
            f"Conflicting authentication headers found: '{TRITON_ACCESS_HEADER}' and "
            f"'{grpc_header_name}' contain different tokens. Please use consistent tokens."
        )

    return http_token or grpc_token


def create_auth_headers(access_token: Optional[str], protocol: str) -> Dict[str, str]:
    """Create protocol-aware authentication headers for Triton Inference Server.

    This function generates the appropriate authentication headers based on the
    protocol being used. Triton Server requires different header formats for
    HTTP and gRPC protocols:

    - HTTP: {"triton-access-token": "token-value"}
    - gRPC: {"triton-grpc-protocol-triton-access-token": "token-value"}

    Args:
        access_token: The access token to include in headers
        protocol: The protocol being used ("http" or "grpc")

    Returns:
        Dictionary containing the appropriate authentication headers

    Raises:
        ValueError: If protocol is not "http" or "grpc"
        ValueError: If access_token is None or empty

    Examples:
        >>> create_auth_headers("my-token", "http")
        {"triton-access-token": "my-token"}

        >>> create_auth_headers("my-token", "grpc")
        {"triton-grpc-protocol-triton-access-token": "my-token"}
    """
    if not access_token:
        raise ValueError("access_token cannot be None or empty")

    protocol = protocol.lower()
    if protocol not in ("http", "grpc"):
        raise ValueError(f"Invalid protocol '{protocol}'. Must be 'http' or 'grpc'")

    if protocol == "http":
        return {TRITON_ACCESS_HEADER: access_token}
    else:  # grpc
        return {f"triton-grpc-protocol-{TRITON_ACCESS_HEADER}": access_token}
