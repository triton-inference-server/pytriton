#!/usr/bin/env python3
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
"""Functional tests for warmup functionality with real Triton instances"""

import contextlib
import logging
import os
import socket
import tempfile
import time
from typing import Dict, List

import numpy as np
import pytest

from pytriton.client import ModelClient
from pytriton.client.exceptions import PyTritonClientTimeoutError
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.model_config.common import ModelWarmup, WarmupInput
from pytriton.triton import Triton, TritonConfig

_LOGGER = logging.getLogger(__name__)
_SMALL_TIMEOUT = 0.1
_GARGANTUAN_TIMEOUT = 15.0

# Global storage for warmup call tracking
WARMUP_CALLS = []


def reset_warmup_calls():
    """Reset the global warmup call tracker."""
    global WARMUP_CALLS
    WARMUP_CALLS = []


def record_warmup_call(inputs: Dict[str, np.ndarray]):
    """Record a warmup call for later validation."""
    global WARMUP_CALLS
    WARMUP_CALLS.append({name: array.copy() for name, array in inputs.items()})


@pytest.fixture(scope="function")
def find_free_ports():
    """Fixture to find free ports for grpc, http, and metrics"""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as grpc:
        grpc.bind(("", 0))
        grpc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as http:
            http.bind(("", 0))
            http.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as metrics:
                metrics.bind(("", 0))
                metrics.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                yield {
                    "grpc_port": grpc.getsockname()[1],
                    "http_port": http.getsockname()[1],
                    "metrics_port": metrics.getsockname()[1],
                }


class TritonWarmupInstance:
    """Context manager to hold Triton instance with warmup configuration."""

    def __init__(
        self, grpc_port, http_port, metrics_port, model_name, infer_function, inputs, outputs, warmup_config=None
    ):
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.metrics_port = metrics_port
        self.model_name = model_name
        self.config = TritonConfig(http_port=http_port, grpc_port=grpc_port, metrics_port=metrics_port)
        self.infer_function = infer_function
        self.grpc_url = f"grpc://localhost:{self.grpc_port}"
        self.http_url = f"http://localhost:{self.http_port}"
        self.inputs = inputs
        self.outputs = outputs
        self.warmup_config = warmup_config

    def __enter__(self):
        try:
            _LOGGER.info("Checking if Triton server is already running.")
            ModelClient(
                self.grpc_url,
                self.model_name,
                init_timeout_s=_SMALL_TIMEOUT,
                inference_timeout_s=_SMALL_TIMEOUT,
                lazy_init=False,
            )
            message = "Triton server already running."
            _LOGGER.error(message)
            raise RuntimeError(message)
        except PyTritonClientTimeoutError:
            _LOGGER.debug("Triton server not running.")
            pass

        # Reset warmup call tracker before starting Triton
        reset_warmup_calls()

        self.triton = Triton(config=self.config)
        _LOGGER.debug("Binding %s model.", self.model_name)

        # Create model config with warmup if provided
        model_config = ModelConfig()
        if self.warmup_config:
            model_config.model_warmup = self.warmup_config

        self.triton.bind(
            model_name=self.model_name,
            infer_func=self.infer_function,
            inputs=self.inputs,
            outputs=self.outputs,
            config=model_config,
            strict=True,
        )
        _LOGGER.info("Running Triton server with warmup.")
        self.triton.run()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _LOGGER.debug("Triton server stopping.")
        self.triton.stop()
        _LOGGER.debug("Checking if Triton server is still running.")
        while True:
            try:
                with ModelClient(
                    self.http_url,
                    self.model_name,
                    init_timeout_s=_SMALL_TIMEOUT,
                    inference_timeout_s=_SMALL_TIMEOUT,
                    lazy_init=False,
                ) as client:
                    _LOGGER.info("Triton server still running. %s", client.model_config)
            except PyTritonClientTimeoutError:
                _LOGGER.debug("Triton server not running.")
                break
            _LOGGER.debug("Triton server still alive, so sleeping for %ss.", _SMALL_TIMEOUT)
            time.sleep(_SMALL_TIMEOUT)
        _LOGGER.info("Triton server stopped.")


def create_warmup_tracking_function(expected_input_names: List[str]):
    """Create an inference function that tracks warmup calls."""

    @batch
    def _warmup_tracking_infer(**inputs):
        _LOGGER.debug("Inference called with inputs: %s", {k: v.shape for k, v in inputs.items()})

        # Record this call for warmup validation
        record_warmup_call(inputs)

        # Generate appropriate outputs based on inputs
        outputs = {}
        for name, data in inputs.items():
            if name == "float_input":
                outputs["float_output"] = data * 2.0  # Simple transformation
            elif name == "int_input":
                outputs["int_output"] = data + 1  # Simple transformation
            elif name == "mixed_float":
                outputs["mixed_float_out"] = data
            elif name == "mixed_int":
                outputs["mixed_int_out"] = data
            else:
                # Default: echo the input as output
                outputs[name + "_output"] = data

        return outputs

    return _warmup_tracking_infer


def test_basic_warmup_functionality(find_free_ports):
    """Test basic warmup functionality with simple float tensors."""

    # Define warmup configuration
    warmup_config = [
        ModelWarmup(
            name="basic_warmup",
            batch_size=2,
            inputs={
                "float_input": WarmupInput(dtype=np.float32, shape=(3,), zero_data=True),
                "int_input": WarmupInput(dtype=np.int64, shape=(2,), random_data=True),
            },
            count=2,
        )
    ]

    # Create inference function
    infer_fn = create_warmup_tracking_function(["float_input", "int_input"])

    # Define model inputs and outputs
    inputs = [
        Tensor(name="float_input", dtype=np.float32, shape=(-1,)),
        Tensor(name="int_input", dtype=np.int64, shape=(-1,)),
    ]
    outputs = [
        Tensor(name="float_output", dtype=np.float32, shape=(-1,)),
        Tensor(name="int_output", dtype=np.int64, shape=(-1,)),
    ]

    with TritonWarmupInstance(
        **find_free_ports,
        model_name="WarmupTestModel",
        infer_function=infer_fn,
        inputs=inputs,
        outputs=outputs,
        warmup_config=warmup_config,
    ) as triton_instance:
        # Verify warmup calls were made
        _LOGGER.info("Recorded %d warmup calls", len(WARMUP_CALLS))
        assert len(WARMUP_CALLS) == 2, f"Expected 2 warmup calls, got {len(WARMUP_CALLS)}"

        # Verify warmup call structure
        for i, call in enumerate(WARMUP_CALLS):
            _LOGGER.info("Warmup call %d: %s", i, call.keys())
            assert "float_input" in call
            assert "int_input" in call

            # Verify shapes (should include batch dimension)
            assert call["float_input"].shape == (2, 3), f"Expected (2, 3), got {call['float_input'].shape}"
            assert call["int_input"].shape == (2, 2), f"Expected (2, 2), got {call['int_input'].shape}"

            # Verify dtypes
            assert call["float_input"].dtype == np.float32
            assert call["int_input"].dtype == np.int64

            # Verify zero data for float_input
            assert np.all(call["float_input"] == 0), "float_input should be zero data"

        # Test that normal inference still works after warmup
        with ModelClient(
            triton_instance.grpc_url,
            triton_instance.model_name,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
        ) as client:
            # Make a normal inference call
            result = client.infer_sample(
                float_input=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                int_input=np.array([10, 20], dtype=np.int64),
            )

            # Verify normal inference works
            assert "float_output" in result
            assert "int_output" in result
            assert result["float_output"].shape == (3,)
            assert result["int_output"].shape == (2,)


def test_multiple_warmup_configurations(find_free_ports):
    """Test multiple warmup configurations with different batch sizes."""

    # Define multiple warmup configurations
    warmup_config = [
        ModelWarmup(
            name="small_batch_warmup",
            batch_size=1,
            inputs={
                "mixed_float": WarmupInput(dtype=np.float32, shape=(2,), zero_data=True),
                "mixed_int": WarmupInput(dtype=np.int32, shape=(1,), random_data=True),
            },
            count=2,
        ),
        ModelWarmup(
            name="large_batch_warmup",
            batch_size=4,
            inputs={
                "mixed_float": WarmupInput(dtype=np.float32, shape=(2,), random_data=True),
                "mixed_int": WarmupInput(dtype=np.int32, shape=(1,), zero_data=True),
            },
            count=1,
        ),
    ]

    # Create inference function
    infer_fn = create_warmup_tracking_function(["mixed_float", "mixed_int"])

    # Define model inputs and outputs
    inputs = [
        Tensor(name="mixed_float", dtype=np.float32, shape=(-1,)),
        Tensor(name="mixed_int", dtype=np.int32, shape=(-1,)),
    ]
    outputs = [
        Tensor(name="mixed_float_out", dtype=np.float32, shape=(-1,)),
        Tensor(name="mixed_int_out", dtype=np.int32, shape=(-1,)),
    ]

    with TritonWarmupInstance(
        **find_free_ports,
        model_name="MultiWarmupModel",
        infer_function=infer_fn,
        inputs=inputs,
        outputs=outputs,
        warmup_config=warmup_config,
    ) as triton_instance:
        # Verify total warmup calls (2 + 1 = 3)
        _LOGGER.info("Recorded %d warmup calls", len(WARMUP_CALLS))
        assert len(WARMUP_CALLS) == 3, f"Expected 3 warmup calls, got {len(WARMUP_CALLS)}"

        # Verify first two calls (batch_size=1)
        for i in range(2):
            call = WARMUP_CALLS[i]
            assert call["mixed_float"].shape == (1, 2), f"Call {i}: Expected (1, 2), got {call['mixed_float'].shape}"
            assert call["mixed_int"].shape == (1, 1), f"Call {i}: Expected (1, 1), got {call['mixed_int'].shape}"
            # First config: float is zero, int is random
            assert np.all(call["mixed_float"] == 0), f"Call {i}: mixed_float should be zero"

        # Verify third call (batch_size=4)
        call = WARMUP_CALLS[2]
        assert call["mixed_float"].shape == (4, 2), f"Call 2: Expected (4, 2), got {call['mixed_float'].shape}"
        assert call["mixed_int"].shape == (4, 1), f"Call 2: Expected (4, 1), got {call['mixed_int'].shape}"
        # Second config: float is random, int is zero
        assert np.all(call["mixed_int"] == 0), "Call 2: mixed_int should be zero"

        # Test normal inference still works
        with ModelClient(
            triton_instance.grpc_url,
            triton_instance.model_name,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
        ) as client:
            result = client.infer_sample(
                mixed_float=np.array([1.5, 2.5], dtype=np.float32),
                mixed_int=np.array([42], dtype=np.int32),
            )
            assert "mixed_float_out" in result
            assert "mixed_int_out" in result


def test_no_warmup_configuration(find_free_ports):
    """Test that models work normally without warmup configuration."""

    # Create inference function
    infer_fn = create_warmup_tracking_function(["simple_input"])

    # Define model inputs and outputs
    inputs = [Tensor(name="simple_input", dtype=np.float32, shape=(-1,))]
    outputs = [Tensor(name="simple_input_output", dtype=np.float32, shape=(-1,))]

    with TritonWarmupInstance(
        **find_free_ports,
        model_name="NoWarmupModel",
        infer_function=infer_fn,
        inputs=inputs,
        outputs=outputs,
        warmup_config=None,  # No warmup
    ) as triton_instance:
        # Verify no warmup calls were made
        _LOGGER.info("Recorded %d warmup calls", len(WARMUP_CALLS))
        assert len(WARMUP_CALLS) == 0, f"Expected 0 warmup calls, got {len(WARMUP_CALLS)}"

        # Test that normal inference works
        with ModelClient(
            triton_instance.grpc_url,
            triton_instance.model_name,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
        ) as client:
            result = client.infer_sample(simple_input=np.array([1.0, 2.0], dtype=np.float32))
            assert "simple_input_output" in result


def test_complex_tensor_types_warmup(find_free_ports):
    """Test warmup with various tensor types and shapes."""

    # Define warmup with complex tensor types
    warmup_config = [
        ModelWarmup(
            name="complex_types_warmup",
            batch_size=2,
            inputs={
                "float16_input": WarmupInput(dtype=np.float16, shape=(3, 4), random_data=True),
                "uint8_input": WarmupInput(dtype=np.uint8, shape=(2,), zero_data=True),
                "bool_input": WarmupInput(dtype=np.bool_, shape=(1,), random_data=True),
            },
            count=1,
        )
    ]

    # Create inference function
    infer_fn = create_warmup_tracking_function(["float16_input", "uint8_input", "bool_input"])

    # Define model inputs and outputs
    inputs = [
        Tensor(name="float16_input", dtype=np.float16, shape=(-1, -1)),
        Tensor(name="uint8_input", dtype=np.uint8, shape=(-1,)),
        Tensor(name="bool_input", dtype=np.bool_, shape=(-1,)),
    ]
    outputs = [
        Tensor(name="float16_input_output", dtype=np.float16, shape=(-1, -1)),
        Tensor(name="uint8_input_output", dtype=np.uint8, shape=(-1,)),
        Tensor(name="bool_input_output", dtype=np.bool_, shape=(-1,)),
    ]

    with TritonWarmupInstance(
        **find_free_ports,
        model_name="ComplexTypesModel",
        infer_function=infer_fn,
        inputs=inputs,
        outputs=outputs,
        warmup_config=warmup_config,
    ) as triton_instance:
        # Verify warmup call
        assert len(WARMUP_CALLS) == 1, f"Expected 1 warmup call, got {len(WARMUP_CALLS)}"

        call = WARMUP_CALLS[0]
        _LOGGER.info("Complex types warmup call: %s", call.keys())

        # Verify shapes and dtypes
        assert call["float16_input"].shape == (2, 3, 4)
        assert call["float16_input"].dtype == np.float16

        assert call["uint8_input"].shape == (2, 2)
        assert call["uint8_input"].dtype == np.uint8
        assert np.all(call["uint8_input"] == 0)  # zero_data

        assert call["bool_input"].shape == (2, 1)
        assert call["bool_input"].dtype == np.bool_

        # Test normal inference
        with ModelClient(
            triton_instance.grpc_url,
            triton_instance.model_name,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
        ) as client:
            result = client.infer_sample(
                float16_input=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16),
                uint8_input=np.array([10, 20], dtype=np.uint8),
                bool_input=np.array([True], dtype=np.bool_),
            )
            assert len(result) == 3  # Should have all three outputs


def test_llm_style_warmup_with_all_types(find_free_ports):
    """Test LLM-style warmup with all warmup types and exact content validation."""

    # Create temporary directory for generated data files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate LLM-style token ID data on the fly
        expected_token_ids = np.array(
            [
                [15496, 995, 11, 428, 318, 257, 1332, 13],  # "Hello world, this is a test."
                [464, 2068, 6520, 12621, 625, 262, 7613, 0],  # "The quick brown fox over the lazy" + padding
            ],
            dtype=np.int64,
        )

        expected_attention_mask = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],  # All tokens are real
                [1, 1, 1, 1, 1, 1, 1, 0],  # Last token is padding
            ],
            dtype=np.int64,
        )

        expected_position_ids = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],  # Sequential positions
                [0, 1, 2, 3, 4, 5, 6, 7],  # Sequential positions
            ],
            dtype=np.int64,
        )

        # Save data files to temporary directory
        token_ids_file = os.path.join(temp_dir, "token_ids.npy")
        attention_mask_file = os.path.join(temp_dir, "attention_mask.npy")
        position_ids_file = os.path.join(temp_dir, "position_ids.npy")

        np.save(token_ids_file, expected_token_ids)
        np.save(attention_mask_file, expected_attention_mask)
        np.save(position_ids_file, expected_position_ids)

        # Define comprehensive warmup configuration testing all three types
        warmup_config = [
            ModelWarmup(
                name="llm_input_data_file_warmup",
                batch_size=2,
                inputs={
                    "token_ids": WarmupInput(dtype=np.int64, shape=(8,), input_data_file=token_ids_file),
                    "attention_mask": WarmupInput(dtype=np.int64, shape=(8,), input_data_file=attention_mask_file),
                    "position_ids": WarmupInput(dtype=np.int64, shape=(8,), input_data_file=position_ids_file),
                },
                count=1,
            ),
            ModelWarmup(
                name="llm_zero_data_warmup",
                batch_size=1,
                inputs={
                    "token_ids": WarmupInput(dtype=np.int64, shape=(8,), zero_data=True),
                    "attention_mask": WarmupInput(dtype=np.int64, shape=(8,), zero_data=True),
                    "position_ids": WarmupInput(dtype=np.int64, shape=(8,), zero_data=True),
                },
                count=1,
            ),
            ModelWarmup(
                name="llm_random_data_warmup",
                batch_size=3,
                inputs={
                    "token_ids": WarmupInput(dtype=np.int64, shape=(8,), random_data=True),
                    "attention_mask": WarmupInput(dtype=np.int64, shape=(8,), random_data=True),
                    "position_ids": WarmupInput(dtype=np.int64, shape=(8,), random_data=True),
                },
                count=2,
            ),
        ]

        # Create LLM inference function with content validation
        @batch
        def llm_infer(**inputs):
            _LOGGER.debug("LLM inference called with inputs: %s", {k: v.shape for k, v in inputs.items()})

            # Record this call for warmup validation
            record_warmup_call(inputs)

            # Validate tensor contents and types
            assert "token_ids" in inputs
            assert "attention_mask" in inputs
            assert "position_ids" in inputs

            token_ids = inputs["token_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = inputs["position_ids"]

            # Verify all are int64 tensors
            assert token_ids.dtype == np.int64, f"Expected int64, got {token_ids.dtype}"
            assert attention_mask.dtype == np.int64, f"Expected int64, got {attention_mask.dtype}"
            assert position_ids.dtype == np.int64, f"Expected int64, got {position_ids.dtype}"

            # Verify shapes are consistent
            batch_size = token_ids.shape[0]
            seq_length = token_ids.shape[1]
            assert attention_mask.shape == (
                batch_size,
                seq_length,
            ), f"Attention mask shape mismatch: {attention_mask.shape}"
            assert position_ids.shape == (batch_size, seq_length), f"Position IDs shape mismatch: {position_ids.shape}"

            # Generate LLM-style outputs (logits for next token prediction)
            vocab_size = 50257  # GPT-2 vocabulary size
            next_token_logits = np.random.random((batch_size, vocab_size)).astype(np.float32)

            # Generate embeddings (typical transformer output)
            hidden_size = 768  # GPT-2 base hidden size
            last_hidden_state = np.random.random((batch_size, seq_length, hidden_size)).astype(np.float32)

            return {"next_token_logits": next_token_logits, "last_hidden_state": last_hidden_state}

        # Define model inputs and outputs
        inputs = [
            Tensor(name="token_ids", dtype=np.int64, shape=(-1,)),
            Tensor(name="attention_mask", dtype=np.int64, shape=(-1,)),
            Tensor(name="position_ids", dtype=np.int64, shape=(-1,)),
        ]
        outputs = [
            Tensor(name="next_token_logits", dtype=np.float32, shape=(-1,)),
            Tensor(name="last_hidden_state", dtype=np.float32, shape=(-1, -1)),
        ]

        with TritonWarmupInstance(
            **find_free_ports,
            model_name="LLMWarmupModel",
            infer_function=llm_infer,
            inputs=inputs,
            outputs=outputs,
            warmup_config=warmup_config,
        ) as triton_instance:
            # Verify total warmup calls (1 + 1 + 2 = 4)
            _LOGGER.info("Recorded %d LLM warmup calls", len(WARMUP_CALLS))
            assert len(WARMUP_CALLS) == 4, f"Expected 4 warmup calls, got {len(WARMUP_CALLS)}"

            # Validate first call (input_data_file warmup)
            call_0 = WARMUP_CALLS[0]
            assert call_0["token_ids"].shape == (2, 8)
            assert call_0["attention_mask"].shape == (2, 8)
            assert call_0["position_ids"].shape == (2, 8)

            # Verify exact content matches expected data from files
            assert np.array_equal(call_0["token_ids"], expected_token_ids), "Token IDs don't match expected content"
            assert np.array_equal(call_0["attention_mask"], expected_attention_mask), (
                "Attention mask doesn't match expected content"
            )
            assert np.array_equal(call_0["position_ids"], expected_position_ids), (
                "Position IDs don't match expected content"
            )

            _LOGGER.info("✓ input_data_file warmup: Exact content validation passed")

            # Validate second call (zero_data warmup)
            call_1 = WARMUP_CALLS[1]
            assert call_1["token_ids"].shape == (1, 8)
            assert call_1["attention_mask"].shape == (1, 8)
            assert call_1["position_ids"].shape == (1, 8)

            # Verify all zeros
            assert np.all(call_1["token_ids"] == 0), "Token IDs should be all zeros"
            assert np.all(call_1["attention_mask"] == 0), "Attention mask should be all zeros"
            assert np.all(call_1["position_ids"] == 0), "Position IDs should be all zeros"

            _LOGGER.info("✓ zero_data warmup: Content validation passed")

            # Validate third and fourth calls (random_data warmup)
            for i, call_idx in enumerate([2, 3]):
                call = WARMUP_CALLS[call_idx]
                assert call["token_ids"].shape == (3, 8), f"Call {call_idx}: Token IDs shape mismatch"
                assert call["attention_mask"].shape == (3, 8), f"Call {call_idx}: Attention mask shape mismatch"
                assert call["position_ids"].shape == (3, 8), f"Call {call_idx}: Position IDs shape mismatch"

                # Verify data is not all zeros (random should generate non-zero values)
                assert not np.all(call["token_ids"] == 0), f"Call {call_idx}: Token IDs should not be all zeros"
                # Note: attention mask and position_ids could be all zeros by chance, so we don't check them

                _LOGGER.info("✓ random_data warmup call %d: Content validation passed", i + 1)

            # Test that normal LLM inference still works after warmup
            with ModelClient(
                triton_instance.grpc_url,
                triton_instance.model_name,
                init_timeout_s=_GARGANTUAN_TIMEOUT,
                inference_timeout_s=_GARGANTUAN_TIMEOUT,
            ) as client:
                # Make a normal LLM inference call
                test_token_ids = np.array([101, 2023, 2003, 1037, 3231], dtype=np.int64)  # "this is a test"
                test_attention_mask = np.array([1, 1, 1, 1, 1], dtype=np.int64)
                test_position_ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)

                result = client.infer_sample(
                    token_ids=test_token_ids,
                    attention_mask=test_attention_mask,
                    position_ids=test_position_ids,
                )

                # Verify normal inference outputs
                assert "next_token_logits" in result
                assert "last_hidden_state" in result
                assert result["next_token_logits"].shape == (50257,), (
                    f"Logits shape: {result['next_token_logits'].shape}"
                )
                assert result["last_hidden_state"].shape == (
                    5,
                    768,
                ), f"Hidden state shape: {result['last_hidden_state'].shape}"

                _LOGGER.info("✓ Normal LLM inference after warmup: Success")


def test_llm_int64_edge_cases_warmup(find_free_ports):
    """Test LLM warmup with edge cases for int64 tensors like large vocab IDs."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create edge case token IDs with large vocabulary values
        large_vocab_tokens = np.array(
            [
                [50256, 50255, 32000, 65535, 100000, 0, 1, 50256],  # Large vocab boundary values
            ],
            dtype=np.int64,
        )

        large_vocab_file = os.path.join(temp_dir, "large_vocab_tokens.npy")
        np.save(large_vocab_file, large_vocab_tokens)

        # Test with edge case warmup
        warmup_config = [
            ModelWarmup(
                name="llm_edge_case_warmup",
                batch_size=1,
                inputs={
                    "input_ids": WarmupInput(dtype=np.int64, shape=(8,), input_data_file=large_vocab_file),
                },
                count=1,
            )
        ]

        @batch
        def edge_case_llm_infer(**inputs):
            record_warmup_call(inputs)

            input_ids = inputs["input_ids"]

            # Verify we can handle large int64 values correctly
            assert input_ids.dtype == np.int64
            assert np.max(input_ids) <= 100000, f"Max value: {np.max(input_ids)}"
            assert np.min(input_ids) >= 0, f"Min value: {np.min(input_ids)}"

            # Simple echo output
            return {"output_ids": input_ids}

        inputs = [Tensor(name="input_ids", dtype=np.int64, shape=(-1,))]
        outputs = [Tensor(name="output_ids", dtype=np.int64, shape=(-1,))]

        with TritonWarmupInstance(
            **find_free_ports,
            model_name="LLMEdgeCaseModel",
            infer_function=edge_case_llm_infer,
            inputs=inputs,
            outputs=outputs,
            warmup_config=warmup_config,
        ) as _:
            # Verify warmup call
            assert len(WARMUP_CALLS) == 1
            call = WARMUP_CALLS[0]

            # Verify exact large vocabulary tokens were passed
            assert np.array_equal(call["input_ids"], large_vocab_tokens), "Large vocab tokens don't match"

            _LOGGER.info("✓ LLM edge case warmup: Large int64 vocabulary validation passed")


def test_comprehensive_int64_warmup_patterns(find_free_ports):
    """Test comprehensive int64 patterns for different LLM use cases."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create different int64 patterns common in LLM inference

        # Pattern 1: Negative values (sometimes used for special tokens)
        special_tokens = np.array(
            [
                [-1, -100, 0, 1, 2],  # Special tokens with negatives
            ],
            dtype=np.int64,
        )

        # Pattern 2: Very large values (large vocabulary)
        large_vocab = np.array(
            [
                [2147483647, 1073741824, 536870912, 268435456, 134217728],  # Near int32 max
            ],
            dtype=np.int64,
        )

        # Pattern 3: Sequential patterns (position encodings)
        positions = np.array(
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
            ],
            dtype=np.int64,
        )

        # Save all patterns
        special_file = os.path.join(temp_dir, "special_tokens.npy")
        large_vocab_file = os.path.join(temp_dir, "large_vocab.npy")
        positions_file = os.path.join(temp_dir, "positions.npy")

        np.save(special_file, special_tokens)
        np.save(large_vocab_file, large_vocab)
        np.save(positions_file, positions)

        # Test with mixed warmup configurations
        warmup_config = [
            ModelWarmup(
                name="special_tokens_warmup",
                batch_size=1,
                inputs={
                    "special_ids": WarmupInput(dtype=np.int64, shape=(5,), input_data_file=special_file),
                },
                count=1,
            ),
            ModelWarmup(
                name="large_vocab_warmup",
                batch_size=1,
                inputs={
                    "vocab_ids": WarmupInput(dtype=np.int64, shape=(5,), input_data_file=large_vocab_file),
                },
                count=1,
            ),
            ModelWarmup(
                name="position_warmup",
                batch_size=3,
                inputs={
                    "position_ids": WarmupInput(dtype=np.int64, shape=(5,), input_data_file=positions_file),
                },
                count=1,
            ),
            # Test random data generation for various int64 ranges
            ModelWarmup(
                name="random_int64_patterns",
                batch_size=2,
                inputs={
                    "random_tokens": WarmupInput(dtype=np.int64, shape=(10,), random_data=True),
                    "zero_padding": WarmupInput(dtype=np.int64, shape=(3,), zero_data=True),
                },
                count=1,
            ),
        ]

        # Comprehensive validation function
        @batch
        def comprehensive_int64_infer(**inputs):
            record_warmup_call(inputs)

            # Validate all inputs are int64
            for name, tensor in inputs.items():
                assert tensor.dtype == np.int64, f"{name} should be int64, got {tensor.dtype}"

                # Validate no NaN or inf values (shouldn't happen with int64, but good to check)
                assert np.all(np.isfinite(tensor)), f"{name} contains non-finite values"

                _LOGGER.debug(
                    "Validated %s: shape=%s, dtype=%s, range=[%s, %s]",
                    name,
                    tensor.shape,
                    tensor.dtype,
                    np.min(tensor),
                    np.max(tensor),
                )

            # Return simple outputs
            outputs = {}
            for name, tensor in inputs.items():
                outputs[name + "_validated"] = tensor
            return outputs

        # Dynamic input/output generation based on warmup config
        all_input_names = set()
        for warmup in warmup_config:
            all_input_names.update(warmup.inputs.keys())

        inputs = [Tensor(name=name, dtype=np.int64, shape=(-1,)) for name in sorted(all_input_names)]
        outputs = [Tensor(name=name + "_validated", dtype=np.int64, shape=(-1,)) for name in sorted(all_input_names)]

        with TritonWarmupInstance(
            **find_free_ports,
            model_name="ComprehensiveInt64Model",
            infer_function=comprehensive_int64_infer,
            inputs=inputs,
            outputs=outputs,
            warmup_config=warmup_config,
        ) as triton_instance:
            # Verify all warmup calls (4 total)
            assert len(WARMUP_CALLS) == 4, f"Expected 4 warmup calls, got {len(WARMUP_CALLS)}"

            # Validate special tokens warmup
            special_call = WARMUP_CALLS[0]
            assert "special_ids" in special_call
            assert np.array_equal(special_call["special_ids"], special_tokens), "Special tokens mismatch"
            assert np.min(special_call["special_ids"]) == -100, "Should contain negative values"
            _LOGGER.info("✓ Special tokens with negative values validated")

            # Validate large vocabulary warmup
            large_vocab_call = WARMUP_CALLS[1]
            assert "vocab_ids" in large_vocab_call
            assert np.array_equal(large_vocab_call["vocab_ids"], large_vocab), "Large vocab mismatch"
            assert np.max(large_vocab_call["vocab_ids"]) == 2147483647, "Should contain large int32 boundary values"
            _LOGGER.info("✓ Large vocabulary int64 values validated")

            # Validate position IDs warmup
            position_call = WARMUP_CALLS[2]
            assert "position_ids" in position_call
            assert np.array_equal(position_call["position_ids"], positions), "Position IDs mismatch"
            assert position_call["position_ids"].shape == (3, 5), "Position batch shape mismatch"
            _LOGGER.info("✓ Sequential position patterns validated")

            # Validate random data warmup
            random_call = WARMUP_CALLS[3]
            assert "random_tokens" in random_call
            assert "zero_padding" in random_call

            assert random_call["random_tokens"].shape == (2, 10), "Random tokens shape mismatch"
            assert random_call["zero_padding"].shape == (2, 3), "Zero padding shape mismatch"
            assert np.all(random_call["zero_padding"] == 0), "Zero padding should be all zeros"

            # Random tokens should have some variation (very unlikely to be all the same)
            random_tokens = random_call["random_tokens"]
            assert len(np.unique(random_tokens)) > 1, "Random tokens should have variation"
            _LOGGER.info("✓ Random int64 generation and zero padding validated")

            # Test normal inference with edge case values
            with ModelClient(
                triton_instance.grpc_url,
                triton_instance.model_name,
                init_timeout_s=_GARGANTUAN_TIMEOUT,
                inference_timeout_s=_GARGANTUAN_TIMEOUT,
            ) as client:
                # Test with mixed edge case inputs
                result = client.infer_sample(
                    special_ids=np.array([-1, 0, 1, 2, 3], dtype=np.int64),
                    vocab_ids=np.array([50000, 100000, 200000, 300000, 400000], dtype=np.int64),
                    position_ids=np.array([0, 1, 2, 3, 4], dtype=np.int64),
                    random_tokens=np.array([42] * 10, dtype=np.int64),
                    zero_padding=np.array([0, 0, 0], dtype=np.int64),
                )

                # Verify all outputs exist and have correct dtypes
                expected_outputs = {
                    "special_ids_validated",
                    "vocab_ids_validated",
                    "position_ids_validated",
                    "random_tokens_validated",
                    "zero_padding_validated",
                }
                assert set(result.keys()) == expected_outputs, (
                    f"Output mismatch: {set(result.keys())} vs {expected_outputs}"
                )

                for output_name, output_tensor in result.items():
                    assert output_tensor.dtype == np.int64, f"{output_name} should be int64"

                _LOGGER.info("✓ Comprehensive int64 edge case inference validated")
