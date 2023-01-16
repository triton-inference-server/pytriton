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
"""Inference decorators tests."""
import numpy as np
import pytest

from pytriton.constants import TRITON_CONTEXT_FIELD_NAME
from pytriton.decorators import (
    TritonContext,
    batch,
    fill_optionals,
    first_value,
    group_by_keys,
    group_by_values,
    pad_batch,
    sample,
    triton_context,
)
from pytriton.exceptions import PytritonValidationError
from pytriton.model_config import DynamicBatcher
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.inference_handler import triton_context_inject

input_requests = [
    {"b": np.array([[1, 2]]), "a": np.array([[1, 9]])},
    {"b": np.array([[3, 4]])},
    {"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1, 1], [1, 1]])},
    {"b": np.array([[1, 2], [1, 2]]), "a": np.array([[2, 4], [2, 4]])},
    {"b": np.array([[1, 2], [1, 2], [9, 9]]), "a": np.array([[1, 1], [1, 1], [1, 1]])},
    {"a": np.array([[1, 1], [1, 1], [1, 1]])},
]

input_requests_for_sample = [{"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1], [1]])}]

input_requests_for_batching = [
    {"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1], [1]])},
    {"b": np.array([[1, 2], [1, 2], [11, 12]]), "a": np.array([[1], [1], [1]])},
    {"b": np.array([[1, 2]]), "a": np.array([[1]])},
]

input_requests_for_groupby_values = [
    {"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1], [1]])},
    {"b": np.array([[1, 2], [1, 2], [11, 12]]), "a": np.array([[1], [1], [1]])},
    {"b": np.array([[1, 2]]), "a": np.array([[1]])},
    {"b": np.array([[5, 6]]), "a": np.array([[2]])},
    {"b": np.array([[7, 2], [4, 2], [1, 122]]), "a": np.array([[2], [2], [2]])},
]

input_batch_with_params = {"b": np.array([[1, 2], [1, 2], [9, 9]]), "a": np.array([[1], [1], [1]])}


def _prepare_context_for_input(inputs):
    a_input = inputs[0]["a"]
    b_input = inputs[0]["b"]

    a_spec = TensorSpec("a", a_input.shape, a_input.dtype)
    b_spec = TensorSpec("b", b_input.shape, b_input.dtype)

    config = TritonModelConfig("a", inputs=[a_spec, b_spec], outputs=[a_spec, b_spec])
    context = TritonContext(config)
    return context


def test_batch():
    @batch
    def batched_fun(**inputs):
        assert isinstance(inputs, dict) and "a" in inputs and "b" in inputs
        assert inputs["a"].shape == (6, 1)
        assert inputs["b"].shape == (6, 2)

        return {"a": inputs["a"] * 2, "b": inputs["b"] * 3}

    results = batched_fun(input_requests_for_batching)

    for input, output in zip(input_requests_for_batching, results):
        assert np.all(input["a"] * 2 == output["a"]) and np.all(input["b"] * 3 == output["b"])


def test_batch_output_list():
    context = _prepare_context_for_input(input_requests_for_batching)

    @batch
    def batched_fun(**inputs):
        assert isinstance(inputs, dict) and "a" in inputs and "b" in inputs
        assert inputs["a"].shape == (6, 1)
        assert inputs["b"].shape == (6, 2)

        return [inputs["a"] * 2, inputs["b"] * 3]

    batched_fun.__triton_context__ = context
    results = batched_fun(input_requests_for_batching)

    for input, output in zip(input_requests_for_batching, results):
        assert np.all(input["a"] * 2 == output["a"]) and np.all(input["b"] * 3 == output["b"])


def test_sample():
    @sample
    def sample_fun(**inputs):
        assert isinstance(inputs, dict) and "a" in inputs and "b" in inputs
        return {"a": inputs["a"] * 2, "b": inputs["b"] * 3}

    results = sample_fun(input_requests_for_sample)

    for input, output in zip(input_requests_for_batching, results):
        assert np.all(input["a"] * 2 == output["a"]) and np.all(input["b"] * 3 == output["b"])


def test_sample_output_list():
    context = _prepare_context_for_input(input_requests_for_sample)

    @sample
    def sample1(**inputs):
        assert isinstance(inputs, dict) and "a" in inputs and "b" in inputs
        return [inputs["a"] * 2, inputs["b"] * 3]

    sample1.__triton_context__ = context
    results = sample1(input_requests_for_sample)

    for input, output in zip(input_requests_for_sample, results):
        assert np.all(input["a"] * 2 == output["a"]) and np.all(input["b"] * 3 == output["b"])


def test_pad_batch():
    @pad_batch
    def padded_fun(**inputs):
        assert "a" in inputs and "b" in inputs
        assert inputs["a"].shape[0] == 4 and inputs["b"].shape[0] == 4
        assert np.all(inputs["a"] == np.array([[1], [1], [1], [1]])) and np.all(
            inputs["b"] == np.array([[1, 2], [1, 2], [9, 9], [9, 9]])
        )
        return inputs

    config = TritonModelConfig("MyModel", max_batch_size=8, batcher=DynamicBatcher(preferred_batch_size=[2, 4, 6]))
    config.batcher.preferred_batch_size = [2, 4, 6]
    padded_fun.__triton_context__ = TritonContext(config)
    results = padded_fun(**(input_batch_with_params.copy()))
    assert results["a"].shape[0] == 4 and results["b"].shape[0] == 4


def test_pad_batch_no_preffered_batch_size():
    @pad_batch
    def padded_fun(**inputs):
        assert "a" in inputs and "b" in inputs
        assert inputs["a"].shape[0] == 8 and inputs["b"].shape[0] == 8
        assert np.all(inputs["a"] == np.array([[1], [1], [1], [1], [1], [1], [1], [1]])) and np.all(
            inputs["b"] == np.array([[1, 2], [1, 2], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9], [9, 9]])
        )

        return inputs

    config = TritonModelConfig("MyModel", max_batch_size=8)
    padded_fun.__triton_context__ = TritonContext(config)
    results = padded_fun(**(input_batch_with_params.copy()))
    assert results["a"].shape[0] == config.max_batch_size and results["b"].shape[0] == config.max_batch_size


def test_flatten():
    @first_value("a")
    def flatten_fun(**inputs):
        assert "a" in inputs and "b" in inputs
        assert isinstance(inputs["a"], int) and inputs["a"] == 1
        return {"b": inputs["b"]}

    results = flatten_fun(**(input_batch_with_params.copy()))
    assert len(results.keys()) == 1 and "b" in results
    assert np.all(results["b"] == input_batch_with_params["b"])


def test_group_by_keys():
    @group_by_keys
    def groupby_keys_fun(inputs):
        for req1, req2 in zip(inputs, inputs[1:]):
            assert req1.keys() == req2.keys()
        k = len(inputs[0].keys())
        results = [{key: inp[key] * k for key in inp} for inp in inputs]
        return results

    results = groupby_keys_fun(input_requests)
    for req, res in zip(input_requests, results):
        assert req.keys() == res.keys()
        for key in req:
            assert np.all(req[key] * len(req.keys()) == res[key])


def test_group_by_values():
    @group_by_values("a")
    def groupby_vals_fun(inputs):
        k = inputs[0]["a"].flatten()[0]
        results = [{"a": inp["a"], "b": inp["b"] * k} for inp in inputs]
        return results

    results = groupby_vals_fun(input_requests_for_groupby_values)
    for req, res in zip(input_requests_for_groupby_values, results):
        assert req.keys() == res.keys()
        assert np.all(req["b"] * req["a"].flatten()[0] == res["b"])


def test_fill_optionals():
    @fill_optionals(a=np.array([-1, -2]), b=np.array((-5, -6)))
    def fill_fun(inputs):
        for req in inputs:
            assert "a" in req and "b" in req
            assert req["a"].shape[0] == req["b"].shape[0]
        assert np.all(inputs[1]["a"] == np.array([[-1, -2], [-1, -2]]))
        assert np.all(inputs[-1]["b"] == np.array([[-5, -6], [-5, -6], [-5, -6]]))
        return inputs

    results = fill_fun(input_requests)
    assert len(results) == len(input_requests)


def test_triton_context():
    @triton_context
    def fun_with_tr_context(inputs, **kwargs):
        assert kwargs.get("triton_context") == "Context"
        assert "a" in inputs
        assert np.all(inputs["a"] == np.array([1, 2]))
        return inputs

    fun_with_tr_context.__triton_context__ = "Context"
    res = fun_with_tr_context({"a": np.array([1, 2])})
    assert "a" in res
    assert np.all(res["a"] == np.array([1, 2]))

    def fun_without_tr_context(inputs, **kwargs):
        assert "triton_context" not in kwargs
        assert "a" in inputs
        assert np.all(inputs["a"] == np.array([1, 2]))
        return inputs

    fun_without_tr_context.__triton_context__ = "Context"
    res = fun_without_tr_context({"a": np.array([1, 2])})
    assert "a" in res
    assert np.all(res["a"] == np.array([1, 2]))


def test_triton_context_not_set():
    @triton_context
    def fun_without_tr_context(inputs, **kwargs):
        pytest.fail("Should not get here. Should raise error before.")

    try:
        _ = fun_without_tr_context({"a": np.array([1, 2])})
        pytest.fail("Error should me raised")
    except PytritonValidationError as ex:
        assert "Wrapped function or object must bound with triton to get  __triton_context__" in ex.message
    except Exception as ex:
        raise RuntimeError("PytritonValidationError should be raised") from ex
        pytest.fail("PytritonValidationError should be raised")


def test_inject_and_acquire_triton_context():
    @triton_context
    class A:
        def __init__(self, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == "Context"

    class B:
        @triton_context
        def fun(self, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == "Context"

    class C:
        @triton_context
        def __call__(self, *args, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == "Context"

    @triton_context
    def fun(**kwargs):
        assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == "Context"

    caller1 = triton_context_inject("Context")(A)
    caller2 = triton_context_inject("Context")(B().fun)
    caller3 = triton_context_inject("Context")(C())
    caller4 = triton_context_inject("Context")(fun)

    caller1()
    caller2()
    caller3()
    caller4()
