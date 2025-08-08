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
"""Tests for pytriton.triton security features and TritonSecurityConfig.

This module contains tests for server-side security functionality:
- TritonSecurityConfig class and its validation
- Security constants (TRITON_ACCESS_HEADER, VALID_TRITON_ENDPOINTS, etc.)
- Token management and endpoint protection integration
- Server configuration with security restrictions
"""

from unittest.mock import MagicMock, patch

import pytest

from pytriton.exceptions import PyTritonValidationError
from pytriton.triton import (
    DEFAULT_PROTECTED_ENDPOINTS,
    TRITON_ACCESS_HEADER,
    VALID_TRITON_ENDPOINTS,
    Triton,
    TritonSecurityConfig,
)


class TestEndpointValidation:
    """Test endpoint name validation functionality."""

    def test_default_protected_endpoints_valid(self):
        """Test that DEFAULT_PROTECTED_ENDPOINTS contains only valid endpoints."""
        # Should not raise any exception
        TritonSecurityConfig._validate_endpoint_names(DEFAULT_PROTECTED_ENDPOINTS)


class TestTritonTokenManagement:
    """Test token-based access restriction functionality."""

    @patch("pytriton.triton.secrets.token_urlsafe")
    def test_explicit_token_used(self, mock_token_urlsafe):
        """Test that explicit token is used when provided."""
        mock_token_urlsafe.return_value = "generated-token"
        explicit_token = "my-explicit-token"
        security_config = TritonSecurityConfig(access_token=explicit_token)

        with patch("pytriton.triton.Triton._initialize_server"):
            triton = Triton(security_config=security_config)

        assert triton.get_access_token() == explicit_token
        # token_urlsafe should not be called when explicit token provided
        mock_token_urlsafe.assert_not_called()

    def test_custom_restricted_endpoints(self):
        """Test that custom restricted endpoints are validated and stored."""
        custom_endpoints = ["health", "metadata", "statistics"]
        security_config = TritonSecurityConfig(restricted_endpoints=custom_endpoints)

        with patch("pytriton.triton.Triton._initialize_server"):
            triton = Triton(security_config=security_config)

        assert triton._security_config.restricted_endpoints == custom_endpoints
        assert triton._security_config.access_token is not None

    def test_invalid_restricted_endpoints_rejected(self):
        """Test that invalid restricted endpoints are rejected."""
        invalid_endpoints = ["health", "invalid-endpoint"]
        with pytest.raises(PyTritonValidationError) as exc_info:
            TritonSecurityConfig(restricted_endpoints=invalid_endpoints)
        assert "Invalid endpoint names: invalid-endpoint" in str(exc_info.value)

    def test_default_endpoints_used_when_none_provided(self):
        """Test that default protected endpoints are used when none provided."""
        with patch("pytriton.triton.Triton._initialize_server"):
            triton = Triton()

        assert triton._security_config.restricted_endpoints == DEFAULT_PROTECTED_ENDPOINTS.copy()


class TestTritonConfigurationFlow:
    """Test the configuration flow from Triton to TritonServerConfig."""

    @patch("pytriton.triton.Triton._initialize_server")
    def test_restriction_parameters_set_in_server_config(self, mock_init_server):
        """Test that restriction parameters are properly set in TritonServerConfig."""
        access_token = "test-token-123"
        restricted_endpoints = ["health", "metadata"]
        security_config = TritonSecurityConfig(access_token=access_token, restricted_endpoints=restricted_endpoints)

        triton = Triton(security_config=security_config)

        # Mock workspace to avoid file system operations
        mock_workspace = MagicMock()
        mock_workspace.model_store_path.as_posix.return_value = "/mock/model/store"

        # Call the method that should configure restrictions
        triton._prepare_triton_config(mock_workspace)

        # Check that both HTTP and gRPC restrictions are configured
        expected_restriction = f"health,metadata:{TRITON_ACCESS_HEADER}=test-token-123"

        assert triton._triton_server_config["grpc-restricted-protocol"] == expected_restriction
        assert triton._triton_server_config["http-restricted-api"] == expected_restriction

    @patch("pytriton.triton.Triton._initialize_server")
    def test_cli_args_generation_includes_restrictions(self, mock_init_server):
        """Test that CLI arguments include restriction parameters."""
        access_token = "cli-test-token"
        restricted_endpoints = ["statistics", "trace"]
        security_config = TritonSecurityConfig(access_token=access_token, restricted_endpoints=restricted_endpoints)

        triton = Triton(security_config=security_config)

        # Mock workspace
        mock_workspace = MagicMock()
        mock_workspace.model_store_path.as_posix.return_value = "/mock/model/store"

        # Configure the server
        triton._prepare_triton_config(mock_workspace)

        # Generate CLI arguments
        cli_args = triton._triton_server_config.to_args_list()
        cli_string = triton._triton_server_config.to_cli_string()

        expected_restriction = f"statistics,trace:{TRITON_ACCESS_HEADER}=cli-test-token"

        # Check that restriction arguments are in the CLI args
        assert f"--grpc-restricted-protocol={expected_restriction}" in cli_string
        assert f"--http-restricted-api={expected_restriction}" in cli_string

        # Check args list format
        assert "--grpc-restricted-protocol" in cli_args
        assert expected_restriction in cli_args
        assert "--http-restricted-api" in cli_args

    def test_get_access_token_returns_string(self):
        """Test that get_access_token always returns a string."""
        security_config = TritonSecurityConfig(access_token="test-token")

        with patch("pytriton.triton.Triton._initialize_server"):
            triton = Triton(security_config=security_config)

        token = triton.get_access_token()
        assert isinstance(token, str)
        assert token == "test-token"


class TestTritonSecurityConfig:
    """Test TritonSecurityConfig class."""

    def test_default_security_config(self):
        """Test that TritonSecurityConfig has correct default values."""
        config = TritonSecurityConfig()

        assert isinstance(config.access_token, str) and len(config.access_token) > 0
        assert config.restricted_endpoints == DEFAULT_PROTECTED_ENDPOINTS

    def test_security_config_with_values(self):
        """Test that TritonSecurityConfig accepts values."""
        token = "test-token-123"
        endpoints = ["statistics", "trace"]

        config = TritonSecurityConfig(access_token=token, restricted_endpoints=endpoints)

        assert config.access_token == token
        assert config.restricted_endpoints == endpoints


class TestEmptyRestrictionsHandling:
    """Test handling of empty restricted_endpoints for unrestricted behavior."""

    def test_empty_list_means_no_restrictions(self):
        """Test that empty list explicitly disables all restrictions."""
        security_config = TritonSecurityConfig(access_token="test-token", restricted_endpoints=[])

        assert security_config.access_token == "test-token"
        assert security_config.restricted_endpoints == []

    def test_none_means_use_defaults(self):
        """Test that None uses default protected endpoints."""
        security_config = TritonSecurityConfig(access_token="test-token", restricted_endpoints=None)

        assert security_config.access_token == "test-token"
        assert security_config.restricted_endpoints == DEFAULT_PROTECTED_ENDPOINTS

    def test_default_behavior_uses_defaults(self):
        """Test that omitting restricted_endpoints uses defaults."""
        security_config = TritonSecurityConfig(access_token="test-token")

        assert security_config.access_token == "test-token"
        assert security_config.restricted_endpoints == DEFAULT_PROTECTED_ENDPOINTS

    @patch("pytriton.triton.Triton._initialize_server")
    def test_empty_restrictions_no_server_config_set(self, mock_init_server):
        """Test that empty restrictions don't set server restriction config."""
        security_config = TritonSecurityConfig(access_token="test-token", restricted_endpoints=[])

        triton = Triton(security_config=security_config)

        # Mock workspace to avoid file system operations
        mock_workspace = MagicMock()
        mock_workspace.model_store_path.as_posix.return_value = "/mock/model/store"

        # Call the method that should configure restrictions
        triton._prepare_triton_config(mock_workspace)

        # Check that NO restrictions are configured when empty list provided
        assert "grpc-restricted-protocol" not in triton._triton_server_config
        assert "http-restricted-api" not in triton._triton_server_config

    @patch("pytriton.triton.Triton._initialize_server")
    def test_none_restrictions_sets_default_server_config(self, mock_init_server):
        """Test that None restrictions set default server restriction config."""
        security_config = TritonSecurityConfig(
            access_token="test-token",
            restricted_endpoints=None,  # Should use defaults
        )

        triton = Triton(security_config=security_config)

        # Mock workspace to avoid file system operations
        mock_workspace = MagicMock()
        mock_workspace.model_store_path.as_posix.return_value = "/mock/model/store"

        # Call the method that should configure restrictions
        triton._prepare_triton_config(mock_workspace)

        # Check that restrictions ARE configured with defaults
        expected_endpoints = ",".join(DEFAULT_PROTECTED_ENDPOINTS)
        expected_restriction = f"{expected_endpoints}:{TRITON_ACCESS_HEADER}=test-token"

        assert triton._triton_server_config["grpc-restricted-protocol"] == expected_restriction
        assert triton._triton_server_config["http-restricted-api"] == expected_restriction

    @patch("pytriton.triton.Triton._initialize_server")
    def test_triton_unrestricted_behavior_restoration(self, mock_init_server):
        """Test complete workflow for restoring unrestricted behavior."""
        # This is the key use case: restore previous unrestricted behavior
        security_config = TritonSecurityConfig(restricted_endpoints=[])

        triton = Triton(security_config=security_config)

        # Verify token is not generated
        assert triton.get_access_token() is None

        # Verify no endpoints are restricted
        assert triton._security_config.restricted_endpoints == []

        # Mock workspace and prepare config
        mock_workspace = MagicMock()
        mock_workspace.model_store_path.as_posix.return_value = "/mock/model/store"
        triton._prepare_triton_config(mock_workspace)

        # Verify no restrictions are applied to server config
        assert "grpc-restricted-protocol" not in triton._triton_server_config
        assert "http-restricted-api" not in triton._triton_server_config


class TestTritonSecurityConfigConstants:
    """Test the constants defined in the triton module."""

    def test_triton_access_header_format(self):
        """Test that TRITON_ACCESS_HEADER has correct format."""
        assert TRITON_ACCESS_HEADER == "triton-access-token"

    def test_valid_endpoints_contains_expected(self):
        """Test that VALID_TRITON_ENDPOINTS contains expected endpoints."""
        expected_endpoints = {
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
        assert VALID_TRITON_ENDPOINTS == expected_endpoints

    def test_default_protected_endpoints_subset_of_valid(self):
        """Test that default protected endpoints are subset of valid endpoints."""
        assert set(DEFAULT_PROTECTED_ENDPOINTS).issubset(VALID_TRITON_ENDPOINTS)

    def test_default_protected_endpoints_expected_values(self):
        """Test that default protected endpoints contain expected values."""
        expected = [
            "shared-memory",
            "model-repository",
            "statistics",
            "trace",
            "logging",
        ]
        assert DEFAULT_PROTECTED_ENDPOINTS == expected


class TestTritonSecurityConfigAdvanced:
    """Test advanced TritonSecurityConfig functionality."""

    def test_none_endpoints_uses_defaults(self):
        """Test that None endpoints uses default protected endpoints."""
        config = TritonSecurityConfig(restricted_endpoints=None)
        assert config.restricted_endpoints == DEFAULT_PROTECTED_ENDPOINTS

    def test_no_token_generation_for_empty_endpoints(self):
        """Test that no token is generated when no endpoints are protected."""
        config = TritonSecurityConfig(restricted_endpoints=[])
        assert config.access_token is None

    def test_explicit_token_preserved(self):
        """Test that explicit token is preserved."""
        explicit_token = "my-explicit-token"
        config = TritonSecurityConfig(access_token=explicit_token)
        assert config.access_token == explicit_token
