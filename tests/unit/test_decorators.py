# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
    ConstantPadder,
    InferenceRequest,
    InferenceRequests,
    InferenceResult,
    InputNames,
    ModelConfigDict,
    TritonContext,
    batch,
    fill_optionals,
    first_value,
    get_model_config,
    group_by_keys,
    group_by_values,
    pad_batch,
    sample,
    triton_context,
)
from pytriton.exceptions import PyTritonBadParameterError, PyTritonRuntimeError, PyTritonValidationError
from pytriton.model_config import DynamicBatcher
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.models.model import _inject_triton_context
from pytriton.proxy.types import Request
from tests.unit.utils import verify_equalness_of_dicts_with_ndarray

input_requests = [
    Request({"b": np.array([[1, 2]]), "a": np.array([[1, 9]])}, {}),
    Request({"b": np.array([[3, 4]])}, {}),
    Request({"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1, 1], [1, 1]])}, {}),
    Request({"b": np.array([[1, 2], [1, 2]]), "a": np.array([[2, 4], [2, 4]])}, {}),
    Request({"b": np.array([[1, 2], [1, 2], [9, 9]]), "a": np.array([[1, 1], [1, 1], [1, 1]])}, {}),
    Request({"a": np.array([[1, 1], [1, 1], [1, 1]])}, {}),
]

input_requests_for_sample = [Request({"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1], [1]])}, {})]

three_request_for_batching = [
    Request({"b": np.array([[7, 5], [8, 6]]), "a": np.array([[1], [1]])}, {}),
    Request({"b": np.array([[1, 2], [1, 2], [11, 12]]), "a": np.array([[1], [1], [1]])}, {}),
    Request({"b": np.array([[1, 2]]), "a": np.array([[1]])}, {}),
]


def _prepare_and_inject_context_with_config(config, fun):
    context = TritonContext()
    context.model_configs[fun] = config
    _inject_triton_context(context, fun)
    return context


def test_get_model_config_key():
    def fn():
        pass

    def fn2():
        pass

    class CallableClass:
        def __call__(self):
            pass

        def method(self):
            pass

    inst = CallableClass()
    inst2 = CallableClass()

    assert ModelConfigDict._get_model_config_key(fn) == str(fn)
    assert ModelConfigDict._get_model_config_key(inst) == str(inst)
    assert ModelConfigDict._get_model_config_key(inst.method) == str(inst.method)
    assert ModelConfigDict._get_model_config_key(inst.__call__) == str(inst)

    config_dict = ModelConfigDict()
    config_dict[fn] = TritonModelConfig(model_name="fn")
    config_dict[fn2] = TritonModelConfig(model_name="fn2")
    assert config_dict[fn] == TritonModelConfig(model_name="fn")
    assert config_dict[fn] != config_dict[fn2]

    config_dict[inst] = TritonModelConfig(model_name="inst")
    config_dict[inst2] = TritonModelConfig(model_name="inst2")
    assert config_dict[inst] == TritonModelConfig(model_name="inst")
    assert config_dict[inst] != config_dict[inst2]

    keys = {fn, fn2, inst, inst2}
    keys1 = set(config_dict.keys())
    keys2 = set(iter(config_dict))
    assert keys == keys1
    assert keys == keys2


def _prepare_context_for_input(inputs, fun):
    a_input = inputs[0]["a"]
    b_input = inputs[0]["b"]

    a_spec = TensorSpec("a", a_input.shape, a_input.dtype)
    b_spec = TensorSpec("b", b_input.shape, b_input.dtype)

    config = TritonModelConfig("a", inputs=[a_spec, b_spec], outputs=[a_spec, b_spec])
    context = TritonContext()
    context.model_configs[fun] = config

    return context


input_batch_with_params = {"b": np.array([[1, 2], [1, 2], [9, 9]]), "a": np.array([[1], [1], [1]])}


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
    _prepare_and_inject_context_with_config(config, padded_fun)
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
    _prepare_and_inject_context_with_config(config, padded_fun)
    results = padded_fun(**(input_batch_with_params.copy()))
    assert results["a"].shape[0] == config.max_batch_size and results["b"].shape[0] == config.max_batch_size


def test_sample():
    @sample
    def sample_fun(**inputs):
        assert isinstance(inputs, dict) and "a" in inputs and "b" in inputs
        return {"a": inputs["a"] * 2, "b": inputs["b"] * 3}

    results = sample_fun(input_requests_for_sample)

    for input, output in zip(three_request_for_batching, results):
        assert np.all(input["a"] * 2 == output["a"]) and np.all(input["b"] * 3 == output["b"])


def test_sample_output_list():
    @sample
    def sample1(**inputs):
        assert isinstance(inputs, dict) and "a" in inputs and "b" in inputs
        return [inputs["a"] * 2, inputs["b"] * 3]

    context = _prepare_context_for_input(input_requests_for_sample, sample1)
    sample1.__triton_context__ = context
    results = sample1(input_requests_for_sample)

    for input, output in zip(input_requests_for_sample, results):
        assert np.all(input["a"] * 2 == output["a"]) and np.all(input["b"] * 3 == output["b"])


_FIRST_VALUE_MODEL_CONFIG = TritonModelConfig(model_name="foo", inputs=[], outputs=[])


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

    _prepare_and_inject_context_with_config(_FIRST_VALUE_MODEL_CONFIG, _fn)

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

    _prepare_and_inject_context_with_config(_FIRST_VALUE_MODEL_CONFIG, _fn)

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

    _prepare_and_inject_context_with_config(_FIRST_VALUE_MODEL_CONFIG, _fn)

    with pytest.raises(PyTritonRuntimeError, match="The values on the .* input are not equal"):
        _fn(a=np.array([[1], [2], [2]]))

    # test disabling strict check
    @first_value("a", strict=False)
    def _fn(**inputs):
        pass

    _prepare_and_inject_context_with_config(_FIRST_VALUE_MODEL_CONFIG, _fn)
    _fn(a=np.array([[1], [2], [2]]))


def test_first_value_raises_on_models_not_supporting_batching():
    @first_value("a")
    def _fn(**inputs):
        pass

    _prepare_and_inject_context_with_config(
        TritonModelConfig(model_name="foo", inputs=[], outputs=[], batching=False), _fn
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
        for key, value in req.items():
            result_value = res[key]
            expected_value = value * len(req.keys())
            assert np.all(expected_value == result_value)


class GroupByValuesTestCase(typing.NamedTuple):
    inference_request: InferenceRequest
    keys: InputNames
    expected: typing.Optional[InferenceRequests] = None
    expected_result: typing.Optional[InferenceResult] = None


_idx1 = "1"
_idx2 = "2"


@pytest.mark.parametrize(
    "inference_request, keys, expected, expected_result",
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
        # using concatenation with _idx variables to avoid string interning
        # https://stackabuse.com/guide-to-string-interning-in-python/
        GroupByValuesTestCase(  # string values
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "s": np.array(
                    [
                        "t" + _idx1,
                        "t" + _idx2,
                        "t" + _idx1,
                        "t" + _idx1,
                        "t" + _idx2,
                        "t" + _idx2,
                        "t" + _idx1,
                        "t" + _idx1,
                        "t" + _idx1,
                        "t" + _idx1,
                    ],
                    dtype=object,
                ),
            },
            keys=["s"],
            expected=(
                {
                    "a": np.array([[1], [1], [1], [2], [2], [2], [2]]),
                    "s": np.array(
                        ["t" + _idx1, "t" + _idx1, "t" + _idx1, "t" + _idx1, "t" + _idx1, "t" + _idx1, "t" + _idx1],
                        dtype=object,
                    ),
                },
                {"a": np.array([[1], [1], [1]]), "s": np.array(["t" + _idx2, "t" + _idx2, "t" + _idx2], dtype=object)},
            ),
        ),
        # using concatenation with _idx variables to avoid string interning
        # https://stackabuse.com/guide-to-string-interning-in-python/
        GroupByValuesTestCase(  # 2d array of string values
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "s": np.array(
                    [
                        ["t" + _idx1, "t" + _idx1],
                        ["t" + _idx2, "t" + _idx2],
                        ["t" + _idx1, "t" + _idx1],
                        ["t" + _idx1, "t" + _idx1],
                        ["t" + _idx2, "t" + _idx2],
                        ["t" + _idx2, "t" + _idx1],
                        ["t" + _idx1, "t" + _idx1],
                        ["t" + _idx1, "t" + _idx1],
                        ["t" + _idx1, "t" + _idx1],
                        ["t" + _idx1, "t" + _idx1],
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
                            ["t" + _idx1, "t" + _idx1],
                            ["t" + _idx1, "t" + _idx1],
                            ["t" + _idx1, "t" + _idx1],
                            ["t" + _idx1, "t" + _idx1],
                            ["t" + _idx1, "t" + _idx1],
                            ["t" + _idx1, "t" + _idx1],
                            ["t" + _idx1, "t" + _idx1],
                        ],
                        dtype=object,
                    ),
                },
                {"a": np.array([[1]]), "s": np.array([["t" + _idx2, "t" + _idx1]], dtype=object)},
                {
                    "a": np.array([[1], [1]]),
                    "s": np.array([["t" + _idx2, "t" + _idx2], ["t" + _idx2, "t" + _idx2]], dtype=object),
                },
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
def test_group_by_values(mocker, inference_request, keys, expected, expected_result):
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


def _expected_test_group_by_values_with_dynamic_axes_on_output():
    expected = np.zeros((10, 16, 4, 4), dtype="int")
    expected[:6, :16, :4, :2] = 1
    expected[6:, :3, :2, :4] = 1
    return expected


@pytest.mark.parametrize(
    "inference_request, keys, expected, expected_result",
    (
        GroupByValuesTestCase(  # output axes: a: (1,), output: (-1,)
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "b": np.array([[7, 5], [8, 6], [1, 2], [1, 2], [11, 12], [1, 2], [5, 6], [7, 2], [4, 2], [1, 122]]),
                "output_length": np.array([[8], [8], [16], [16], [4], [8], [3], [3], [3], [3]]),
            },
            keys=["a"],
            expected_result={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "output": np.block([  # 16 is the max output_length
                    [np.ones((6, 16), dtype="int")],
                    [np.ones((4, 3), dtype="int"), np.zeros((4, 13), dtype="int")],
                ]),
            },
        ),
        GroupByValuesTestCase(  # output axes: a: (1,), output: (-1, -1, -1)
            inference_request={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "b": np.array([[7, 5], [8, 6], [1, 2], [1, 2], [11, 12], [1, 2], [5, 6], [7, 2], [4, 2], [1, 122]]),
                "output_length": np.array([
                    [8, 2, 1],
                    [8, 2, 1],
                    [16, 4, 2],
                    [16, 4, 2],
                    [4, 2, 2],
                    [8, 2, 2],
                    [3, 1, 4],
                    [3, 1, 4],
                    [3, 2, 2],
                    [3, 2, 2],
                ]),
            },
            keys=["a"],
            expected_result={
                "a": np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]]),
                "output": _expected_test_group_by_values_with_dynamic_axes_on_output(),
            },
        ),
    ),
)
def test_group_by_values_with_dynamic_axes_on_output(mocker, inference_request, keys, expected, expected_result):
    @group_by_values(*keys, pad_fn=ConstantPadder(0))
    def _fn(**inputs):
        return {
            "a": inputs["a"],
            "output": np.ones((len(inputs["a"]), *np.max(inputs["output_length"], axis=0).tolist()), dtype="int"),
        }

    result = _fn(**inference_request)
    verify_equalness_of_dicts_with_ndarray(result, expected_result)


def test_group_by_values_with_dynamic_axes_of_bytes_on_output():
    @group_by_values("a", pad_fn=ConstantPadder(0))
    def _fn(**inputs):
        if inputs["a"][0][0] == 1:
            sequences = np.array([
                [b"foo", b"barxxx", b""],
                [b"bar1", b"Loriem ipsum", b"foo"],
                [b"foo", b"barxxx", b""],
                [b"bar1", b"Loriem ipsum", b"foo"],
                [b"foo", b"barxxx", b""],
                [b"bar1", b"Loriem ipsum", b"foo"],
            ])
        else:
            sequences = np.array([
                [b"foo", b"bar", b"", b""],
                [b"1", b"22", b"3", b"4444"],
                [b"foo", b"bar", b"", b""],
                [b"1", b"22", b"3", b"4444"],
            ])

        return {"a": inputs["a"], "output": sequences}

    a = np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]])
    inference_request = {"a": a}
    expected_result = {
        "a": a,
        "output": np.array([
            [b"foo", b"barxxx", b"", b""],
            [b"bar1", b"Loriem ipsum", b"foo", b""],
            [b"foo", b"barxxx", b"", b""],
            [b"bar1", b"Loriem ipsum", b"foo", b""],
            [b"foo", b"barxxx", b"", b""],
            [b"bar1", b"Loriem ipsum", b"foo", b""],
            [b"foo", b"bar", b"", b""],
            [b"1", b"22", b"3", b"4444"],
            [b"foo", b"bar", b"", b""],
            [b"1", b"22", b"3", b"4444"],
        ]),
    }

    result = _fn(**inference_request)
    verify_equalness_of_dicts_with_ndarray(result, expected_result)


def test_group_by_values_with_dynamic_axes_of_unicode_on_output():
    @group_by_values("a", pad_fn=ConstantPadder(0))
    def _fn(**inputs):
        if inputs["a"][0][0] == 1:
            sequences = np.array([
                ["foo", "barxxx", ""],
                ["bar1", "Loriem ipsum", "foo"],
                ["foo", "barxxx", ""],
                ["bar1", "Loriem ipsum", "foo"],
                ["foo", "barxxx", ""],
                ["bar1", "Loriem ipsum", "foo"],
            ])
        else:
            sequences = np.array([
                ["foo", "bar", "", ""],
                ["1", "22", "3", "4444"],
                ["foo", "bar", "", ""],
                ["1", "22", "3", "4444"],
            ])

        return {"a": inputs["a"], "output": sequences}

    a = np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]])
    inference_request = {"a": a}
    expected_result = {
        "a": a,
        "output": np.array([
            ["foo", "barxxx", "", ""],
            ["bar1", "Loriem ipsum", "foo", ""],
            ["foo", "barxxx", "", ""],
            ["bar1", "Loriem ipsum", "foo", ""],
            ["foo", "barxxx", "", ""],
            ["bar1", "Loriem ipsum", "foo", ""],
            ["foo", "bar", "", ""],
            ["1", "22", "3", "4444"],
            ["foo", "bar", "", ""],
            ["1", "22", "3", "4444"],
        ]),
    }

    result = _fn(**inference_request)
    verify_equalness_of_dicts_with_ndarray(result, expected_result)


def test_group_by_values_with_dynamic_axes_of_bytes_as_objects_on_output():
    @group_by_values("a", pad_fn=ConstantPadder(0))
    def _fn(**inputs):
        if inputs["a"][0][0] == 1:
            sequences = np.array(
                [
                    [b"foo", b"barxxx", b""],
                    [b"bar1", b"Loriem ipsum", b"foo"],
                    [b"foo", b"barxxx", b""],
                    [b"bar1", b"Loriem ipsum", b"foo"],
                    [b"foo", b"barxxx", b""],
                    [b"bar1", b"Loriem ipsum", b"foo"],
                ],
                dtype=object,
            )
        else:
            sequences = np.array(
                [
                    [b"foo", b"bar", b"", b""],
                    [b"1", b"22", b"3", b"4444"],
                    [b"foo", b"bar", b"", b""],
                    [b"1", b"22", b"3", b"4444"],
                ],
                dtype=object,
            )

        return {"a": inputs["a"], "output": sequences}

    a = np.array([[1], [1], [1], [1], [1], [1], [2], [2], [2], [2]])
    inference_request = {"a": a}
    expected_result = {
        "a": a,
        "output": np.array(
            [
                [b"foo", b"barxxx", b"", b""],
                [b"bar1", b"Loriem ipsum", b"foo", b""],
                [b"foo", b"barxxx", b"", b""],
                [b"bar1", b"Loriem ipsum", b"foo", b""],
                [b"foo", b"barxxx", b"", b""],
                [b"bar1", b"Loriem ipsum", b"foo", b""],
                [b"foo", b"bar", b"", b""],
                [b"1", b"22", b"3", b"4444"],
                [b"foo", b"bar", b"", b""],
                [b"1", b"22", b"3", b"4444"],
            ],
            dtype=object,
        ),
    }

    result = _fn(**inference_request)
    verify_equalness_of_dicts_with_ndarray(result, expected_result)


def test_group_by_values_raise_error_if_placed_before_batch():
    with pytest.raises(
        PyTritonRuntimeError, match="The @group_by_values decorator must be used after the @batch decorator."
    ):

        @group_by_values("a")
        @batch
        def _fn(**_requests):
            return _requests

        _fn([{"a": np.zeros((1,))}, {"a": np.zeros((1,))}])


def test_fill_optionals_in_instance_callable():
    class MyModel:
        @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
        def __call__(self, inputs):
            for req in inputs:
                assert "a" in req and "b" in req
                assert req["a"].shape[0] == req["b"].shape[0]
            assert np.all(inputs[1]["a"] == np.array([[-1, -2], [-1, -2]]))
            assert np.all(inputs[-1]["b"] == np.array([[-5, -6], [-5, -6], [-5, -6]]))
            return inputs

    model = MyModel()

    _prepare_and_inject_context_with_config(
        TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        ),
        model.__call__,
    )

    results = model(input_requests)
    assert len(results) == len(input_requests)


def test_fill_optionals():
    @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
    def fill_fun(inputs):
        for req in inputs:
            assert "a" in req and "b" in req
            assert req["a"].shape[0] == req["b"].shape[0]
        assert np.all(inputs[1]["a"] == np.array([[-1, -2], [-1, -2]]))
        assert np.all(inputs[-1]["b"] == np.array([[-5, -6], [-5, -6], [-5, -6]]))
        return inputs

    _prepare_and_inject_context_with_config(
        TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        ),
        fill_fun,
    )

    results = fill_fun(input_requests)
    assert len(results) == len(input_requests)


def test_fill_optionals_for_not_batching_models():
    @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    _prepare_and_inject_context_with_config(
        TritonModelConfig(
            model_name="foo",
            batching=False,
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        ),
        infer_fn,
    )

    inputs = [
        Request({"a": np.array([1, 9]), "b": np.array([1, 2])}, {}),
        Request({"b": np.array([3, 4])}, {}),
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

    _prepare_and_inject_context_with_config(
        TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        ),
        infer_fn,
    )

    with pytest.raises(PyTritonBadParameterError, match="Could not use a=.* they are not NumPy arrays"):
        infer_fn(input_requests)


def test_fill_optionals_raise_error_on_dtype_mismatch():
    @fill_optionals(a=np.array([-1, -2]), b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    _prepare_and_inject_context_with_config(
        TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int32), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        ),
        infer_fn,
    )

    with pytest.raises(
        PyTritonBadParameterError, match="Could not use a: dtype=.* have different than input signature dtypes"
    ):
        infer_fn(input_requests)


def test_fill_optionals_raise_error_on_shape_mismatch():
    @fill_optionals(a=np.array([[-1, -2]]), b=np.array([-5, -6]))
    def infer_fn(inputs):
        return inputs

    _prepare_and_inject_context_with_config(
        TritonModelConfig(
            model_name="foo",
            inputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
            outputs=[TensorSpec("a", shape=(2,), dtype=np.int64), TensorSpec("b", shape=(2,), dtype=np.int64)],
        ),
        infer_fn,
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
    context = TritonContext()

    @triton_context
    class A:
        def __init__(self, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == context

        @classmethod
        def __call__(cls, *args, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == context

    class B:
        @triton_context
        def fun(self, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == context

    class C:
        @triton_context
        def __call__(self, *args, **kwargs):
            assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == context

    @triton_context
    def fun(**kwargs):
        assert kwargs.get(TRITON_CONTEXT_FIELD_NAME) == context

    caller1 = _inject_triton_context(context, A)
    caller2 = _inject_triton_context(context, B().fun)
    caller3 = _inject_triton_context(context, C())
    caller4 = _inject_triton_context(context, fun)

    caller1()
    caller2()
    caller3()
    caller4()


def test_get_triton_context_with_decorators_stack():
    """There should be possible to obtain TritonContext from any decorator in wrappers stack"""

    dummy_config = TritonModelConfig("foo")

    @wrapt.decorator
    def my_decorator(wrapped, instance, args, kwargs):
        _config = get_model_config(wrapped, instance)
        assert _config == dummy_config

    @my_decorator
    @batch
    def infer_fn(**kwargs):
        return kwargs

    _prepare_and_inject_context_with_config(dummy_config, infer_fn)
    infer_fn()
