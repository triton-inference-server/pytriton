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
import typing

import numpy as np
import pytest
import wrapt

from pytriton.constants import TRITON_CONTEXT_FIELD_NAME
from pytriton.decorators import (
    InferenceRequest,
    InferenceRequests,
    InputNames,
    TritonContext,
    batch,
    fill_optionals,
    first_value,
    get_triton_context,
    group_by_keys,
    group_by_values,
    pad_batch,
    sample,
    triton_context,
)
from pytriton.exceptions import PyTritonBadParameterError, PyTritonRuntimeError, PyTritonValidationError
from pytriton.model_config import DynamicBatcher
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.inference_handler import triton_context_inject
from tests.unit.utils import verify_equalness_of_dicts_with_ndarray

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


_FIRST_VALUE_TRITON_CONTEXT = TritonContext(model_config=TritonModelConfig(model_name="foo", inputs=[], outputs=[]))


@pytest.mark.parametrize(
    "inputs, keys, expected",
    (
        (  # extract 1st item (scalar) from 1D array
            {"a": np.array([1, 2, 3]), "b": np.array([1, 1, 1])},
            ["b"],
            {"a": np.array([1, 2, 3]), "b": np.int64(1)},
        ),
        (  # extract 1st item (scalar) from 3D array of shape (batch_size, 1, 1)
            {"a": np.array([1, 2, 3]), "b": np.array([[[1]], [[1]], [[1]]])},
            ["b"],
            {"a": np.array([1, 2, 3]), "b": np.int64(1)},
        ),
        (  # extract 1st item (2D) from 3D array of shape != (batch_size, 1, 1)
            {"a": np.array([1, 2, 3]), "b": np.array([[[1], [2]], [[1], [2]], [[1], [2]]])},
            ["b"],
            {"a": np.array([1, 2, 3]), "b": np.array([[1], [2]])},
        ),
        (  # extract 1st item (scalar) from 1D array of strings (objects)
            {"a": np.array([1, 2, 3]), "b": np.array(["val1", "val1"], dtype=object)},
            ["b"],
            {"a": np.array([1, 2, 3]), "b": np.object_("val1")},
        ),
        (  # extract 1st item (scalar) from 3D array of strings (objects) with shape (batch_size, 1, 1)
            {"a": np.array([1, 2, 3]), "b": np.array([[["val1"]], [["val1"]]], dtype=object)},
            ["b"],
            {"a": np.array([1, 2, 3]), "b": np.object_("val1")},
        ),
        (  # do not raise error when key is missing in inputs
            {"a": np.array([1, 2, 3]), "b": np.array([1, 1, 1])},
            ["c"],  # optional name
            {"a": np.array([1, 2, 3]), "b": np.array([1, 1, 1])},
        ),
        (  # extract 1st item (scalar) from 1D array + do not raise error when key is missing in inputs
            {"a": np.array([1, 2, 3]), "b": np.array([1, 1, 1])},
            ["b", "c"],  # optional name
            {"a": np.array([1, 2, 3]), "b": np.int64(1)},
        ),
        (  # extract 1st item (scalar) from 1D array on 2 inputs
            {"a": np.array([2, 2, 2]), "b": np.array([1, 1, 1])},
            ["a", "b"],
            {"a": np.int64(2), "b": np.int64(1)},
        ),
    ),
)
def test_first_value_with_single_request(mocker, inputs, keys, expected):
    """Assume @batch is before decorator"""

    class PassTrough:
        def __call__(self, **_inputs):
            return _inputs

    passtrough = PassTrough()
    spy_passtrough = mocker.spy(passtrough, "__call__")

    @first_value(*keys)
    def _fn(**_inputs):
        return spy_passtrough(**_inputs)

    _fn.__triton_context__ = _FIRST_VALUE_TRITON_CONTEXT

    result = _fn(**inputs)
    verify_equalness_of_dicts_with_ndarray(result, expected)

    for call_args, expected_args in zip(spy_passtrough.call_args_list, [expected]):
        verify_equalness_of_dicts_with_ndarray(call_args.kwargs, expected_args)


@pytest.mark.parametrize(
    "requests, keys, expected",
    (
        (  # single request - extract 1st item (scalar) from 1D array
            [{"a": np.array([1, 2, 3]), "b": np.array([1, 1, 1])}],
            ["b"],
            [{"a": np.array([1, 2, 3]), "b": np.int64(1)}],
        ),
        (  # multiple requests - extract 1st item (scalar) from 3D array of shape (batch_size, 1, 1)
            [
                {"a": np.array([1, 2, 3]), "b": np.array([[[1]], [[1]], [[1]]])},
                {"a": np.array([1, 2, 3]), "b": np.array([[[1]], [[1]], [[1]]])},
            ],
            ["b", "optional"],
            [
                {"a": np.array([1, 2, 3]), "b": np.int64(1)},
                {"a": np.array([1, 2, 3]), "b": np.int64(1)},
            ],
        ),
    ),
)
def test_first_value_with_requests(mocker, requests, keys, expected):
    """Assume no @batch is before decorator"""

    class PassTrough:
        def __call__(self, _requests):
            return _requests

    passtrough = PassTrough()
    spy_passtrough = mocker.spy(passtrough, "__call__")

    @first_value(*keys)
    def _fn(_requests):
        return spy_passtrough(_requests)

    _fn.__triton_context__ = _FIRST_VALUE_TRITON_CONTEXT

    results = _fn(requests)
    assert len(results) == len(expected)
    for result, expected_request in zip(results, expected):
        verify_equalness_of_dicts_with_ndarray(result, expected_request)

    for call_args, expected_requests in zip(spy_passtrough.call_args_list, [expected]):
        called_requests, *_ = call_args.args
        for called_request, expected_request in zip(called_requests, expected_requests):
            verify_equalness_of_dicts_with_ndarray(called_request, expected_request)


def test_first_value_raises_on_special_key():

    with pytest.raises(PyTritonBadParameterError, match="not allowed as keys for @first_value wrapper."):

        @first_value("__triton_context__")
        def _fn(**inputs):
            pass


def test_first_value_raises_on_not_equal_values():
    @first_value("a")
    def _fn(**inputs):
        pass

    _fn.__triton_context__ = _FIRST_VALUE_TRITON_CONTEXT

    with pytest.raises(PyTritonRuntimeError, match="The values on the .* input are not equal"):
        _fn(a=np.array([[1], [2], [2]]))

    # test disabling strict check
    @first_value("a", strict=False)
    def _fn(**inputs):
        pass

    _fn.__triton_context__ = _FIRST_VALUE_TRITON_CONTEXT
    _fn(a=np.array([[1], [2], [2]]))


def test_first_value_raises_on_models_not_supporting_batching():
    @first_value("a")
    def _fn(**inputs):
        pass

    _fn.__triton_context__ = TritonContext(
        model_config=TritonModelConfig(model_name="foo", inputs=[], outputs=[], batching=False)
    )

    with pytest.raises(
        PyTritonRuntimeError, match="The @first_value decorator can only be used with models that support batching."
    ):
        _fn(a=np.array([[1], [2], [2]]))


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


class GroupByValuesTestCase(typing.NamedTuple):
    inference_request: InferenceRequest
    keys: InputNames
    expected: InferenceRequests


@pytest.mark.parametrize(
    "inference_request, keys, expected",
    (
        GroupByValuesTestCase(
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "b": np.array([[7, 5], [8, 6], [1, 2], [1, 2], [11, 12], [1, 2], [5, 6], [7, 2], [4, 2], [1, 122]]),
            },
            keys=["a"],
            expected=(
                {
                    "a": np.array([[1], [1], [1], [1], [1], [1]]),
                    "b": np.array([[7, 5], [8, 6], [1, 2], [1, 2], [11, 12], [1, 2]]),
                },
                {"a": np.array([[2], [2], [2], [2]]), "b": np.array([[5, 6], [7, 2], [4, 2], [1, 122]])},
            ),
        ),
        GroupByValuesTestCase(  # string values
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "s": np.array(["t1", "t2", "t1", "t1", "t2", "t2", "t1", "t1", "t1", "t1"], dtype=object),
            },
            keys=["s"],
            expected=(
                {
                    "a": np.array([[1], [1], [1], [2], [2], [2], [2]]),
                    "s": np.array(["t1", "t1", "t1", "t1", "t1", "t1", "t1"], dtype=object),
                },
                {"a": np.array([[1], [1], [1]]), "s": np.array(["t2", "t2", "t2"], dtype=object)},
            ),
        ),
        GroupByValuesTestCase(  # 2d array of string values
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "s": np.array(
                    [
                        ["t1", "t1"],
                        ["t2", "t2"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                        ["t2", "t2"],
                        ["t2", "t1"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                    ],
                    dtype=object,
                ),
            },
            keys=["s"],
            expected=(
                {
                    "a": np.array([[1], [1], [1], [2], [2], [2], [2]]),
                    "s": np.array(
                        [
                            ["t1", "t1"],
                            ["t1", "t1"],
                            ["t1", "t1"],
                            ["t1", "t1"],
                            ["t1", "t1"],
                            ["t1", "t1"],
                            ["t1", "t1"],
                        ],
                        dtype=object,
                    ),
                },
                {"a": np.array([[1]]), "s": np.array([["t2", "t1"]], dtype=object)},
                {"a": np.array([[1], [1]]), "s": np.array([["t2", "t2"], ["t2", "t2"]], dtype=object)},
            ),
        ),
        GroupByValuesTestCase(  # group by 2 keys
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "s": np.array(
                    [
                        ["t1", "t1"],
                        ["t2", "t2"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                        ["t2", "t2"],
                        ["t2", "t1"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                        ["t1", "t1"],
                    ],
                    dtype=object,
                ),
            },
            keys=["a", "s"],
            expected=(
                {
                    "a": np.array([[1], [1], [1]]),
                    "s": np.array([["t1", "t1"], ["t1", "t1"], ["t1", "t1"]], dtype=object),
                },
                {"a": np.array([[1]]), "s": np.array([["t2", "t1"]], dtype=object)},
                {"a": np.array([[1], [1]]), "s": np.array([["t2", "t2"], ["t2", "t2"]], dtype=object)},
                {
                    "a": np.array([[2], [2], [2], [2]]),
                    "s": np.array([["t1", "t1"], ["t1", "t1"], ["t1", "t1"], ["t1", "t1"]], dtype=object),
                },
            ),
        ),
    ),
)
def test_group_by_values(mocker, inference_request, keys, expected):
    class PassTrough:
        def __call__(self, **inputs):
            return inputs

    passtrough = PassTrough()
    spy_passtrough = mocker.spy(passtrough, "__call__")

    @group_by_values(*keys)
    def _fn(**inputs):
        return spy_passtrough(**inputs)

    result = _fn(**inference_request)
    verify_equalness_of_dicts_with_ndarray(result, inference_request)

    for call_args, expected_request in zip(spy_passtrough.call_args_list, expected):
        called_request = call_args.kwargs
        verify_equalness_of_dicts_with_ndarray(called_request, expected_request)


def test_group_by_values_raise_error_if_placed_before_batch():
    with pytest.raises(
        PyTritonRuntimeError, match="The @group_by_values decorator must be used after the @batch decorator."
    ):

        @group_by_values("a")
        @batch
        def _fn(**_requests):
            return _requests

        _fn([{"a": np.zeros((1,))}, {"a": np.zeros((1,))}])


def test_fill_optionals():
    @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
    def fill_fun(inputs):
        for req in inputs:
            assert "a" in req and "b" in req
            assert req["a"].shape[0] == req["b"].shape[0]
        assert np.all(inputs[1]["a"] == np.array([[-1, -2], [-1, -2]]))
        assert np.all(inputs[-1]["b"] == np.array([[-5, -6], [-5, -6], [-5, -6]]))
        return inputs

    fill_fun.__triton_context__ = TritonContext(
        model_config=TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        )
    )
    results = fill_fun(input_requests)
    assert len(results) == len(input_requests)


def test_fill_optionals_for_not_batching_models():
    @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    infer_fn.__triton_context__ = TritonContext(
        model_config=TritonModelConfig(
            model_name="foo",
            batching=False,
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        )
    )

    inputs = [
        {"a": np.array([1, 9]), "b": np.array([1, 2])},
        {"b": np.array([3, 4])},
    ]
    expected_results = [
        {"a": np.array([1, 9]), "b": np.array([1, 2])},
        {"a": np.array([-1, -2]), "b": np.array([3, 4])},
    ]

    results = infer_fn(inputs)
    for result, expected_result in zip(results, expected_results):
        assert not set(result) ^ set(expected_result)
        for input_name in result:
            np.testing.assert_array_equal(result[input_name], expected_result[input_name])


def test_fill_optionals_raise_on_non_numpy_defaults():
    @fill_optionals(a=1, b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    infer_fn.__triton_context__ = TritonContext(
        model_config=TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        )
    )

    with pytest.raises(PyTritonBadParameterError, match="Could not use a=.* they are not NumPy arrays"):
        infer_fn(input_requests)


def test_fill_optionals_raise_error_on_dtype_mismatch():
    @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    infer_fn.__triton_context__ = TritonContext(
        model_config=TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int32), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        )
    )

    with pytest.raises(
        PyTritonBadParameterError, match="Could not use a: dtype=.* have different than input signature dtypes"
    ):
        infer_fn(input_requests)


def test_fill_optionals_raise_error_on_shape_mismatch():
    @fill_optionals(a=np.array([[-1, -2]]), b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    infer_fn.__triton_context__ = TritonContext(
        model_config=TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        )
    )

    with pytest.raises(
        PyTritonBadParameterError, match="Could not use a: shape=.*have different than input signature shapes"
    ):
        infer_fn(input_requests)


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
    except PyTritonValidationError as ex:
        assert "Wrapped function or object must bound with triton to get  __triton_context__" in ex.message
    except Exception as ex:
        raise RuntimeError("PyTritonValidationError should be raised") from ex
        pytest.fail("PyTritonValidationError should be raised")


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


def test_get_triton_context_with_decorators_stack():
    """There should be possible to obtain TritonContext from any decorator in wrappers stack"""
    dummy_triton_context = TritonContext(model_config=TritonModelConfig("foo"))

    @wrapt.decorator
    def my_decorator(wrapped, instance, args, kwargs):
        _triton_context = get_triton_context(wrapped, instance)
        assert _triton_context == dummy_triton_context

    @my_decorator
    @batch
    def infer_fn(**kwargs):
        return kwargs

    infer_fn.__triton_context__ = dummy_triton_context
    infer_fn()
