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
"""Tests for the pytriton.client.auth module functionality.

This module contains tests for client-specific authentication functions:
- create_auth_headers() - Protocol-aware header generation
- extract_access_token_from_headers() - Token extraction from headers
"""

import pytest

from pytriton.client.auth import create_auth_headers, extract_access_token_from_headers
from pytriton.triton import TRITON_ACCESS_HEADER


class TestExtractAccessTokenFromHeaders:
    """Test the extract_access_token_from_headers function."""

    def test_extract_http_format_token(self):
        """Test extracting token from HTTP format headers."""
        headers = {TRITON_ACCESS_HEADER: "test-token-123"}
        token = extract_access_token_from_headers(headers)
        assert token == "test-token-123"

    def test_extract_grpc_format_token(self):
        """Test extracting token from gRPC format headers."""
        headers = {f"triton-grpc-protocol-{TRITON_ACCESS_HEADER}": "test-token-456"}
        token = extract_access_token_from_headers(headers)
        assert token == "test-token-456"

    def test_extract_http_preferred_over_grpc_same_token(self):
        """Test that HTTP format is preferred when both formats have same token."""
        headers = {TRITON_ACCESS_HEADER: "same-token", f"triton-grpc-protocol-{TRITON_ACCESS_HEADER}": "same-token"}
        token = extract_access_token_from_headers(headers)
        assert token == "same-token"

    def test_extract_conflicting_tokens_raises_error(self):
        """Test that conflicting tokens raise ValueError."""
        headers = {TRITON_ACCESS_HEADER: "http-token", f"triton-grpc-protocol-{TRITON_ACCESS_HEADER}": "grpc-token"}
        with pytest.raises(ValueError, match="Conflicting authentication headers found"):
            extract_access_token_from_headers(headers)

    def test_extract_no_token_found(self):
        """Test extracting from headers with no auth tokens."""
        headers = {"other-header": "other-value"}
        token = extract_access_token_from_headers(headers)
        assert token is None

    def test_extract_empty_headers(self):
        """Test extracting from empty headers dictionary."""
        headers = {}
        token = extract_access_token_from_headers(headers)
        assert token is None

    def test_extract_none_headers(self):
        """Test extracting from None headers."""
        token = extract_access_token_from_headers(None)
        assert token is None

    def test_extract_token_with_mixed_headers(self):
        """Test extracting token when mixed with other headers."""
        headers = {"content-type": "application/json", TRITON_ACCESS_HEADER: "mixed-token", "user-agent": "test-client"}
        token = extract_access_token_from_headers(headers)
        assert token == "mixed-token"


class TestCreateAuthHeaders:
    """Test the create_auth_headers function."""

    def test_http_header_generation(self):
        """Test creating HTTP authentication headers."""
        headers = create_auth_headers("test-token", "http")
        expected = {TRITON_ACCESS_HEADER: "test-token"}
        assert headers == expected

    def test_grpc_header_generation(self):
        """Test creating gRPC authentication headers."""
        headers = create_auth_headers("test-token", "grpc")
        expected = {f"triton-grpc-protocol-{TRITON_ACCESS_HEADER}": "test-token"}
        assert headers == expected

    def test_protocol_case_insensitive(self):
        """Test that protocol parameter is case-insensitive."""
        headers_upper = create_auth_headers("test-token", "HTTP")
        headers_lower = create_auth_headers("test-token", "http")
        assert headers_upper == headers_lower

        headers_upper = create_auth_headers("test-token", "GRPC")
        headers_lower = create_auth_headers("test-token", "grpc")
        assert headers_upper == headers_lower

    def test_invalid_protocol_raises_error(self):
        """Test that invalid protocol raises ValueError."""
        with pytest.raises(ValueError, match="Invalid protocol 'invalid'. Must be 'http' or 'grpc'"):
            create_auth_headers("test-token", "invalid")

    def test_empty_token_raises_error(self):
        """Test that empty token raises ValueError."""
        with pytest.raises(ValueError, match="access_token cannot be None or empty"):
            create_auth_headers("", "http")

    def test_none_token_raises_error(self):
        """Test that None token raises ValueError."""
        with pytest.raises(ValueError, match="access_token cannot be None or empty"):
            create_auth_headers(None, "http")
