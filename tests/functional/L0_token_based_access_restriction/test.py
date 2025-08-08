#!/usr/bin/env python3
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
"""Test of token-based access restriction functionality"""

import argparse
import contextlib
import logging
import sys
import time
from typing import Dict, Optional

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
    "platforms": ["amd64", "arm64"],
}


def identity_infer_fn(input_1) -> Dict[str, object]:
    """Simple identity function for testing."""
    return {"output_1": input_1}


def get_batched_identity_infer_fn():
    """Get the batched version of identity_infer_fn."""
    from pytriton.decorators import batch

    return batch(identity_infer_fn)


@contextlib.contextmanager
def create_test_triton_server(test_name: str, security_config: Optional[object] = None):
    """Create a Triton server for testing with consistent setup and logging."""
    import numpy as np

    from pytriton.check.utils import find_free_port
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    LOGGER.info("[%s] Starting Triton server setup", test_name)

    http_port = find_free_port()
    grpc_port = find_free_port()
    batched_infer_fn = get_batched_identity_infer_fn()

    LOGGER.info("[%s] Using ports - HTTP: %s, gRPC: %s", test_name, http_port, grpc_port)

    triton_kwargs = {
        "config": TritonConfig(
            http_port=http_port,
            grpc_port=grpc_port,
            log_verbose=0,
        )
    }

    if security_config:
        triton_kwargs["security_config"] = security_config
        LOGGER.info("[%s] Security config enabled", test_name)

    with Triton(**triton_kwargs) as triton:
        LOGGER.info("[%s] Binding model", test_name)
        triton.bind(
            model_name="identity",
            infer_func=batched_infer_fn,
            inputs=[Tensor(name="input_1", dtype=np.float32, shape=(-1,))],
            outputs=[Tensor(name="output_1", dtype=np.float32, shape=(-1,))],
            config=ModelConfig(max_batch_size=16),
        )

        LOGGER.info("[%s] Starting server", test_name)
        triton.run()

        # Yield server info for tests
        yield {
            "triton": triton,
            "base_url": f"http://localhost:{http_port}",
            "http_port": http_port,
            "grpc_port": grpc_port,
        }

        triton.stop()

    LOGGER.info("[%s] Server stopped", test_name)


def test_basic_functionality_without_security():
    """Test that basic functionality works without security config."""
    import requests

    test_name = "NO_SECURITY"
    LOGGER.info("[%s] Testing basic functionality without security", test_name)

    with create_test_triton_server(test_name) as server_context:
        base_url = server_context["base_url"]

        LOGGER.info("[%s] Testing health endpoint", test_name)
        response = requests.get(f"{base_url}/v2/health/ready", timeout=10)
        assert response.status_code == 200, f"Health endpoint failed: {response.status_code}"

        LOGGER.info("[%s] Testing model config endpoint", test_name)
        response = requests.get(f"{base_url}/v2/models/identity/config", timeout=10)
        assert response.status_code == 200, f"Model config endpoint failed: {response.status_code}"

        LOGGER.info("[%s] Test passed", test_name)


def test_auto_generated_token():
    """Test token-based access restriction with auto-generated token."""
    import requests

    from pytriton.triton import TritonSecurityConfig

    test_name = "AUTO_TOKEN"
    LOGGER.info("[%s] Testing auto-generated token functionality", test_name)

    # Create security config with auto-generated token
    security_config = TritonSecurityConfig()
    token = security_config.access_token
    LOGGER.info("[%s] Generated token: %s...", test_name, token[:10])

    with create_test_triton_server(test_name, security_config) as server_context:
        base_url = server_context["base_url"]

        LOGGER.info("[%s] Testing protected endpoint without token", test_name)
        response = requests.post(f"{base_url}/v2/repository/index", json={"ready": True}, timeout=10)
        assert response.status_code in [401, 403], f"Expected 401/403, got {response.status_code}"

        LOGGER.info("[%s] Testing protected endpoint with valid token", test_name)
        headers = {"triton-access-token": token}
        response = requests.post(f"{base_url}/v2/repository/index", json={"ready": True}, headers=headers, timeout=10)
        assert response.status_code == 200, f"Repository index with token failed: {response.status_code}"

        LOGGER.info("[%s] Testing non-protected endpoint without token", test_name)
        response = requests.get(f"{base_url}/v2/health/ready", timeout=10)
        assert response.status_code == 200, f"Health endpoint failed: {response.status_code}"

        LOGGER.info("[%s] Testing inference endpoint", test_name)
        test_data = {"inputs": [{"name": "input_1", "datatype": "FP32", "shape": [1, 3], "data": [1.0, 2.0, 3.0]}]}
        response = requests.post(f"{base_url}/v2/models/identity/infer", json=test_data, timeout=10)
        assert response.status_code == 200, f"Inference failed: {response.status_code}"

        LOGGER.info("[%s] Test passed", test_name)


def test_custom_token():
    """Test token-based access restriction with custom token."""
    import requests

    from pytriton.triton import TritonSecurityConfig

    test_name = "CUSTOM_TOKEN"
    custom_token = "my-super-secret-token-12345"
    LOGGER.info("[%s] Testing custom token functionality", test_name)

    security_config = TritonSecurityConfig(access_token=custom_token)

    with create_test_triton_server(test_name, security_config) as server_context:
        base_url = server_context["base_url"]

        LOGGER.info("[%s] Testing with wrong token", test_name)
        wrong_headers = {"triton-access-token": "wrong-token"}
        response = requests.post(
            f"{base_url}/v2/repository/index", json={"ready": True}, headers=wrong_headers, timeout=10
        )
        assert response.status_code in [401, 403], f"Expected 401/403 with wrong token, got {response.status_code}"

        LOGGER.info("[%s] Testing with correct custom token", test_name)
        correct_headers = {"triton-access-token": custom_token}
        response = requests.post(
            f"{base_url}/v2/repository/index", json={"ready": True}, headers=correct_headers, timeout=10
        )
        assert response.status_code == 200, f"Repository index with custom token failed: {response.status_code}"

        LOGGER.info("[%s] Test passed", test_name)


def test_custom_protected_endpoints():
    """Test token-based access restriction with custom protected endpoints."""
    import requests

    from pytriton.triton import TritonSecurityConfig

    test_name = "CUSTOM_ENDPOINTS"
    custom_token = "test-token"
    custom_endpoints = ["statistics", "trace"]  # Only these should be protected
    LOGGER.info("[%s] Testing custom protected endpoints: %s", test_name, custom_endpoints)

    security_config = TritonSecurityConfig(
        access_token=custom_token,
        restricted_endpoints=custom_endpoints,
    )

    with create_test_triton_server(test_name, security_config) as server_context:
        base_url = server_context["base_url"]
        headers = {"triton-access-token": custom_token}

        LOGGER.info("[%s] Testing protected endpoint without token", test_name)
        response = requests.get(f"{base_url}/v2/models/stats", timeout=10)
        assert response.status_code in [401, 403], f"Statistics should be protected, got {response.status_code}"

        LOGGER.info("[%s] Testing protected endpoint with token", test_name)
        response = requests.get(f"{base_url}/v2/models/stats", headers=headers, timeout=10)
        assert response.status_code == 200, f"Statistics with token failed: {response.status_code}"

        LOGGER.info("[%s] Test passed", test_name)


def test_invalid_endpoint_validation():
    """Test that invalid endpoint names are rejected."""
    from pytriton.triton import TritonSecurityConfig

    test_name = "INVALID_ENDPOINTS"
    LOGGER.info("[%s] Testing invalid endpoint validation", test_name)

    try:
        # This should raise an exception due to invalid endpoint name
        TritonSecurityConfig(
            access_token="test-token",
            restricted_endpoints=["invalid-endpoint", "statistics"],
        )
        raise AssertionError("Expected exception for invalid endpoint name")
    except Exception as e:
        assert "Invalid endpoint names" in str(e), f"Unexpected exception: {e}"
        LOGGER.info("[%s] Test passed", test_name)


def test_concurrent_requests():
    """Test concurrent requests with and without tokens."""
    import concurrent.futures

    import requests

    from pytriton.triton import TritonSecurityConfig

    test_name = "CONCURRENT"
    token = "concurrent-test-token"
    LOGGER.info("[%s] Testing concurrent requests", test_name)

    security_config = TritonSecurityConfig(access_token=token)

    with create_test_triton_server(test_name, security_config) as server_context:
        base_url = server_context["base_url"]
        headers = {"triton-access-token": token}

        def make_request(use_token=True):
            request_headers = headers if use_token else {}
            try:
                response = requests.get(f"{base_url}/v2/models/identity/stats", headers=request_headers, timeout=5)
                return response.status_code
            except Exception as e:
                return str(e)

        LOGGER.info("[%s] Making concurrent requests", test_name)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit requests with and without tokens
            future_with_token = [executor.submit(make_request, True) for _ in range(3)]
            future_without_token = [executor.submit(make_request, False) for _ in range(3)]

            # Check results
            for future in future_with_token:
                result = future.result()
                assert result == 200, f"Request with token failed: {result}"

            for future in future_without_token:
                result = future.result()
                assert result in [401, 403], f"Request without token should fail: {result}"

        LOGGER.info("[%s] Test passed", test_name)


def test_unrestricted_behavior_restoration():
    """Test that empty restricted_endpoints restores unrestricted behavior."""
    import requests

    from pytriton.triton import TritonSecurityConfig

    test_name = "UNRESTRICTED"
    LOGGER.info("[%s] Testing unrestricted behavior with empty endpoint list", test_name)

    # Use empty list to explicitly disable all restrictions
    security_config = TritonSecurityConfig(
        access_token="test-token-but-not-used",
        restricted_endpoints=[],  # Empty list = no restrictions
    )

    with create_test_triton_server(test_name, security_config) as server_context:
        base_url = server_context["base_url"]
        triton = server_context["triton"]

        LOGGER.info("[%s] Testing all endpoints work without token", test_name)

        # Health endpoint should work without token
        response = requests.get(f"{base_url}/v2/health/ready", timeout=10)
        assert response.status_code == 200, f"Health endpoint failed: {response.status_code}"

        # Statistics endpoint should work without token (previously protected by default)
        response = requests.get(f"{base_url}/v2/models/stats", timeout=10)
        assert response.status_code == 200, f"Statistics endpoint should be unrestricted: {response.status_code}"

        # Test that token is still generated but not required
        token = triton.get_access_token()
        assert token is not None and len(token) > 0, "Token should still be generated"
        LOGGER.info("[%s] Token generated but ignored: %s...", test_name, token[:10])

        # Test that wrong token also works (because no restrictions are applied)
        wrong_headers = {"triton-access-token": "wrong-token"}
        response = requests.get(f"{base_url}/v2/models/stats", headers=wrong_headers, timeout=10)
        assert response.status_code == 200, f"Statistics with wrong token should still work: {response.status_code}"

        LOGGER.info("[%s] Test passed", test_name)


def test_default_behavior():
    """Test default behavior with security config (should have restrictions)."""
    import requests

    from pytriton.triton import TritonSecurityConfig

    test_name = "DEFAULT_BEHAVIOR"
    LOGGER.info("[%s] Testing default security behavior", test_name)

    # Default behavior - should protect default endpoints
    default_security_config = TritonSecurityConfig(access_token="default-token")

    with create_test_triton_server(test_name, default_security_config) as server_context:
        base_url = server_context["base_url"]

        LOGGER.info("[%s] Testing protected endpoint without token", test_name)
        response = requests.get(f"{base_url}/v2/models/stats", timeout=10)
        assert response.status_code in [401, 403], f"Statistics should be protected by default: {response.status_code}"

        LOGGER.info("[%s] Testing protected endpoint with token", test_name)
        headers = {"triton-access-token": "default-token"}
        response = requests.get(f"{base_url}/v2/models/stats", headers=headers, timeout=10)
        assert response.status_code == 200, f"Statistics with token should work: {response.status_code}"

        LOGGER.info("[%s] Test passed", test_name)


def main():
    """Main test execution function."""
    from pytriton.check.utils import DEFAULT_LOG_FORMAT

    parser = argparse.ArgumentParser(description="Test token-based access restriction functionality")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)

    start_time = time.time()

    # List of all test functions with descriptive names
    test_functions = [
        ("Basic functionality without security", test_basic_functionality_without_security),
        ("Auto-generated token", test_auto_generated_token),
        ("Custom token", test_custom_token),
        ("Custom protected endpoints", test_custom_protected_endpoints),
        ("Invalid endpoint validation", test_invalid_endpoint_validation),
        ("Concurrent requests", test_concurrent_requests),
        ("Unrestricted behavior restoration", test_unrestricted_behavior_restoration),
        ("Default behavior", test_default_behavior),
    ]

    try:
        LOGGER.info("=" * 60)
        LOGGER.info("STARTING TOKEN-BASED ACCESS RESTRICTION TESTS")
        LOGGER.info("=" * 60)

        # Run all test cases
        for test_name, test_func in test_functions:
            LOGGER.info("Running: %s", test_name)
            test_func()
            LOGGER.info("Completed: %s", test_name)
            LOGGER.info("-" * 40)

        elapsed_time = time.time() - start_time
        LOGGER.info("=" * 60)
        LOGGER.info("ALL TOKEN-BASED ACCESS RESTRICTION TESTS PASSED")
        LOGGER.info("Total execution time: %.2f seconds", elapsed_time)
        LOGGER.info("=" * 60)

        if elapsed_time >= args.timeout_s:
            LOGGER.warning("Tests completed but exceeded timeout")
            sys.exit(-2)

    except Exception as e:
        elapsed_time = time.time() - start_time
        LOGGER.error("=" * 60)
        LOGGER.error("TEST FAILED after %.2f seconds: %s", elapsed_time, e)
        LOGGER.error("=" * 60)
        raise


if __name__ == "__main__":
    main()
