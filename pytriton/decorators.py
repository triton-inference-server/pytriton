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
"""Inference callable decorators."""
import collections
import dataclasses
import inspect
import itertools
import operator
import typing
from bisect import bisect_left
from collections.abc import MutableMapping
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import wrapt

from pytriton.constants import TRITON_CONTEXT_FIELD_NAME
from pytriton.exceptions import PyTritonBadParameterError, PyTritonRuntimeError, PyTritonValidationError
from pytriton.model_config.triton_model_config import TritonModelConfig
from pytriton.proxy.data import _serialize_byte_tensor

_WrappedWithWrapper = NamedTuple(
    "WrappedWithWrapper", [("wrapped", Optional[Callable]), ("wrapper", Optional[Callable])]
)


InputNames = typing.List[str]
InferenceRequest = typing.Dict[str, np.ndarray]
InferenceRequests = typing.Union[typing.List[InferenceRequest], typing.Tuple[InferenceRequest, ...]]
InferenceResult = typing.Dict[str, np.ndarray]
InferenceResults = typing.Union[typing.List[InferenceResult], typing.Tuple[InferenceResult, ...]]


def get_inference_request_batch_size(inference_request: InferenceRequest) -> int:
    """Get batch size from triton request.

    Args:
        inference_request (InferenceRequest): Triton request.

    Returns:
        int: Batch size.
    """
    first_input_value = next(iter(inference_request.values()))
    batch_size, *dims = first_input_value.shape
    return batch_size


def _get_wrapt_stack(wrapped) -> List[_WrappedWithWrapper]:
    """Returns stack of wrapped functions with wrappers applied to inference callable."""
    stack = []
    infer_callable = wrapped
    while infer_callable is not None:
        stack.append(_WrappedWithWrapper(infer_callable, getattr(infer_callable, "_self_wrapper", None)))
        infer_callable = getattr(infer_callable, "__wrapped__", None)

    return stack


class ModelConfigDict(MutableMapping):
    """Dictionary for storing model configs for inference callable."""

    def __init__(self):
        """Create ModelConfigDict object."""
        self._data: Dict[str, TritonModelConfig] = {}
        self._keys: List[Callable] = []

    def __getitem__(self, infer_callable: Callable) -> TritonModelConfig:
        """Get model config for inference callable."""
        key = self._get_model_config_key(infer_callable)
        return self._data[key]

    def __setitem__(self, infer_callable: Callable, item: TritonModelConfig):
        """Set model config for inference callable."""
        self._keys.append(infer_callable)
        key = self._get_model_config_key(infer_callable)
        self._data[key] = item

    def __delitem__(self, infer_callable: Callable):
        """Delete model config for inference callable."""
        key = self._get_model_config_key(infer_callable)
        del self._data[key]

    def __len__(self):
        """Get number of inference callable keys."""
        return len(self._data)

    def __iter__(self):
        """Iterate over inference callable keys."""
        return iter(self._keys)

    @staticmethod
    def _get_model_config_key(infer_callable: Callable) -> str:
        """Prepares TritonModelConfig dictionary key for function/callable."""
        dict_key = infer_callable
        if inspect.ismethod(dict_key) and dict_key.__name__ == "__call__":
            dict_key = dict_key.__self__
        return str(dict_key)


@dataclasses.dataclass
class TritonContext:
    """Triton context definition class."""

    model_configs: ModelConfigDict = dataclasses.field(default_factory=ModelConfigDict)


def get_triton_context(wrapped, instance) -> TritonContext:
    """Retrieves triton context from callable.

    It is used in @triton_context to get triton context registered by triton binding in inference callable.
    If you use @triton_context decorator you do not need this function.
    """
    caller = instance or wrapped
    if not hasattr(caller, "__triton_context__"):
        raise PyTritonValidationError("Wrapped function or object must bound with triton to get  __triton_context__")
    return caller.__triton_context__


def get_model_config(wrapped, instance) -> TritonModelConfig:
    """Retrieves instance of TritonModelConfig from callable.

    It is internally used in convert_output function to get output list from model.
    You can use this in custom decorators if you need access to model_config information.
    If you use @triton_context decorator you do not need this function (you can get model_config directly
    from triton_context passing function/callable to dictionary getter).
    """
    return get_triton_context(wrapped, instance).model_configs[wrapped]


def convert_output(
    outputs: Union[Dict, List, Tuple], wrapped=None, instance=None, model_config: Optional[TritonModelConfig] = None
):
    """Converts output from tuple ot list to dictionary.

    It is utility function useful for mapping output list into dictionary of outputs.
    Currently, it is used in @sample and @batch decorators (we assume that user can return list or tuple of outputs
    instead of dictionary if this list matches output list in model config (size and order).
    """
    if isinstance(outputs, dict):
        return outputs
    elif isinstance(outputs, (list, tuple)):
        if model_config is None:
            model_config = get_model_config(wrapped, instance)
        if len(outputs) != len(model_config.outputs):
            raise PyTritonValidationError("Outputs length different than config outputs length")
        outputs = {config_output.name: output for config_output, output in zip(model_config.outputs, outputs)}
        return outputs
    else:
        raise PyTritonValidationError(f"Unsupported output type {type(outputs)}.")


@wrapt.decorator
def sample(wrapped, instance, args, kwargs):
    """Decorator is used for non-batched inputs to convert from one element list of requests to request kwargs.

    Decorator takes first request and convert it into named inputs.
    Useful with non-batching models - instead of one element list of request, we will get named inputs - `kwargs`.
    """
    kwargs.update(args[0][0])
    outputs = wrapped(*args[1:], **kwargs)
    outputs = convert_output(outputs, wrapped, instance)
    return [outputs]


@wrapt.decorator
def batch(wrapped, instance, args, kwargs):
    """Decorator for converting list of request dicts to dict of input batches.

    Converts list of request dicts to dict of input batches.
    It passes **kwargs to inference callable where each named input contains numpy array with batch of requests
    received by Triton server.
    We assume that each request has the same set of keys (you can use group_by_keys decorator before
    using @batch decorator if your requests may have different set of keys).

    Raises:
        PyTritonValidationError: If the requests have different set of keys.
        ValueError: If the output tensors have different than expected batch sizes. Expected batch size is
            calculated as a sum of batch sizes of all requests.
    """
    req_list = args[0]
    input_names = req_list[0].keys()

    for req_dict2 in req_list[1:]:
        if input_names != req_dict2.keys():
            raise PyTritonValidationError("Cannot batch requests with different set of inputs keys")

    inputs = {}
    for model_input in input_names:
        concatenated_input_data = np.concatenate([req[model_input] for req in req_list])
        inputs[model_input] = concatenated_input_data

    args = args[1:]
    new_kwargs = dict(kwargs)
    new_kwargs.update(inputs)
    outputs = wrapped(*args, **new_kwargs)

    def _split_result(_result):
        outputs = convert_output(_result, wrapped, instance)
        output_names = outputs.keys()

        requests_total_batch_size = sum(get_inference_request_batch_size(req) for req in req_list)
        not_matching_tensors_shapes = {
            output_name: output_tensor.shape
            for output_name, output_tensor in outputs.items()
            if output_tensor.shape[0] != requests_total_batch_size
        }
        if not_matching_tensors_shapes:
            raise ValueError(
                f"Received output tensors with different batch sizes: {', '.join(': '.join(map(str, item)) for item in not_matching_tensors_shapes.items())}. "
                f"Expected batch size: {requests_total_batch_size}. "
            )

        out_list = []
        start_idx = 0
        for request in req_list:
            # get batch_size of first input for each request - assume that all inputs have same batch_size
            request_batch_size = get_inference_request_batch_size(request)
            req_output_dict = {}
            for _output_ind, output_name in enumerate(output_names):
                req_output = outputs[output_name][start_idx : start_idx + request_batch_size, ...]
                req_output_dict[output_name] = req_output
            out_list.append(req_output_dict)
            start_idx += request_batch_size
        return out_list

    if inspect.isgenerator(outputs):
        return (_split_result(_result) for _result in outputs)
    else:
        return _split_result(outputs)


def group_by_values(*keys, pad_fn: typing.Optional[typing.Callable[[InferenceRequests], InferenceRequests]] = None):
    """Decorator for grouping requests by values of selected keys.

    This function splits a batch into multiple sub-batches based on the specified keys values and
    calls the decorated function with each sub-batch. This is particularly useful when working with models
    that require dynamic parameters sent by the user.

    For example, given an input of the form:

    ```python
    {"sentences": [b"Sentence1", b"Sentence2", b"Sentence3"], "param1": [1, 1, 2], "param2": [1, 1, 1]}
    ```

    Using @group_by_values("param1", "param2") will split the batch into two sub-batches:

    ```python
    [
        {"sentences": [b"Sentence1", b"Sentence2"], "param1": [1, 1], "param2": [1, 1]},
        {"sentences": [b"Sentence3"], "param1": [2], "param2": [1]}
    ]
    ```

    This decorator should be used after the @batch decorator.

    Example usage:
    ```python
    @batch
    @group_by_values("param1", "param2")
    def infer_fun(**inputs):
        ...
        return outputs
    ```

    Args:
        *keys: List of keys to group by.
        pad_fn: Optional function to pad the batch to the same size before merging again to a single batch.

    Returns:
        The decorator function.
    """

    def value_to_key(value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.object_ or value.dtype.type == np.bytes_:
                return _serialize_byte_tensor(value)
            else:
                return value.tobytes()
        return value

    def _get_sort_key_for_sample(_request, _sample_idx: int):
        return tuple(value_to_key(_request[_key][_sample_idx]) for _key in keys)

    def _group_request(_request: InferenceRequest, _batch_size: int):
        idx_inputs = [(sample_idx, _get_sort_key_for_sample(_request, sample_idx)) for sample_idx in range(_batch_size)]
        idx_inputs.sort(key=operator.itemgetter(1))
        for _, group in itertools.groupby(idx_inputs, key=operator.itemgetter(1)):
            _samples_idxes, _ = zip(*group)
            grouped_request = {input_name: value[_samples_idxes, ...] for input_name, value in _request.items()}
            yield _samples_idxes, grouped_request

    @wrapt.decorator
    def _wrapper(wrapped, instance, args, kwargs):
        wrappers_stack = [
            callable_with_wrapper.wrapper
            for callable_with_wrapper in _get_wrapt_stack(wrapped)
            if callable_with_wrapper.wrapper is not None
        ]
        if batch in wrappers_stack:
            raise PyTritonRuntimeError("The @group_by_values decorator must be used after the @batch decorator.")

        request = {k: v for k, v in kwargs.items() if k not in _SPECIAL_KEYS}
        other_kwargs = {k: v for k, v in kwargs.items() if k in _SPECIAL_KEYS}

        batch_size = get_inference_request_batch_size(request)
        sample_indices_with_interim_result = []
        for sample_indices, _grouped_sub_request in _group_request(request, batch_size):
            interim_result = wrapped(*args, **_grouped_sub_request, **other_kwargs)
            sample_indices_with_interim_result.append((sample_indices, interim_result))

        if pad_fn is not None:
            indices, results = tuple(map(tuple, zip(*sample_indices_with_interim_result)))
            results = pad_fn(results)
            sample_indices_with_interim_result = tuple(zip(indices, results))

        _, first_result_data = sample_indices_with_interim_result[0]
        result = {
            output_name: np.zeros((batch_size,) + data.shape[1:], dtype=data.dtype)
            for output_name, data in first_result_data.items()
        }
        for indices, results in sample_indices_with_interim_result:
            for output_name, data in results.items():
                result[output_name][indices, ...] = data

        return result

    return _wrapper


class ConstantPadder:
    """Padder that pads the given batches with a constant value."""

    def __init__(self, pad_value=0):
        """Initialize the padder.

        Args:
            pad_value (int, optional): Padding value. Defaults to 0.
        """
        self.pad_value = pad_value

    def __call__(self, batches_list: InferenceResults) -> InferenceResults:
        """Pad the given batches with the specified value to pad size enabling further batching to single arrays.

        Args:
            batches_list (List[Dict[str, np.ndarray]]): List of batches to pad.

        Returns:
            List[Dict[str, np.ndarray]]: List of padded batches.

        Raises:
            PyTritonRuntimeError: If the input arrays for a given input name have different dtypes.
        """

        def _get_padded_shape(_batches: List[np.ndarray]) -> Tuple[int, ...]:
            """Get the shape of the padded array without batch axis."""
            return tuple(np.max([batch.shape[1:] for batch in _batches if batch is not None], axis=0))

        def _get_padded_dtype(_batches: List[np.ndarray]) -> np.dtype:
            dtypes = [batch.dtype for batch in _batches if batch is not None]
            result_dtype = dtypes[0]

            if not all(dtype.kind == result_dtype.kind for dtype in dtypes):
                raise PyTritonRuntimeError("All input arrays for given input name must have the same dtype.")

            # for bytes (encoded string) or unicode string need to obtain the max length
            if result_dtype.kind in "SU":
                order_and_kind = result_dtype.str[:2]
                max_len = max([int(dtype.str[2:]) for dtype in dtypes])
                result_dtype = f"{order_and_kind}{max_len}"
            else:
                if not all(dtype == result_dtype for dtype in dtypes):
                    raise PyTritonRuntimeError("All input arrays for given input name must have the same dtype.")

            return np.dtype(result_dtype)

        input_names = list(
            collections.OrderedDict.fromkeys(input_name for batch in batches_list for input_name in batch.keys())
        )
        batches_by_name = {input_name: [batch.get(input_name) for batch in batches_list] for input_name in input_names}
        for input_batches in batches_by_name.values():
            result_shape, result_dtype = _get_padded_shape(input_batches), _get_padded_dtype(input_batches)
            for batch_idx, batch in enumerate(input_batches):
                if batch is not None:
                    input_batches[batch_idx] = np.pad(
                        batch,
                        [(0, 0)] + [(0, b - a) for a, b in zip(batch.shape[1:], result_shape)],
                        mode="constant",
                        constant_values=self.pad_value if result_dtype.kind not in ["S", "U", "O"] else b"",
                    ).astype(result_dtype)

        return [
            {name: batches[batch_idx] for name, batches in batches_by_name.items() if batches[batch_idx] is not None}
            for batch_idx in range(len(batches_list))
        ]


@wrapt.decorator
def group_by_keys(wrapped, instance, args, kwargs):
    """Group by keys.

    Decorator prepares groups of requests with the same set of keys and calls wrapped function
    for each group separately (it is convenient to use this decorator before batching, because the batching decorator
    requires consistent set of inputs as it stacks them into batches).
    """
    inputs = args[0]
    idx_inputs = [(idx, tuple(sorted(input.keys())), input) for idx, input in enumerate(inputs)]
    idx_inputs.sort(key=operator.itemgetter(1))
    idx_groups_res = []
    for _, group in itertools.groupby(idx_inputs, key=operator.itemgetter(1)):
        idx, _key, sample_list = zip(*group)
        args = (list(sample_list),) + args[1:]
        out = wrapped(*args, **kwargs)
        idx_groups_res.extend(zip(idx, out))

    idx_groups_res.sort(key=operator.itemgetter(0))
    res_flat = [r[1] for r in idx_groups_res]
    return res_flat


def fill_optionals(**defaults):
    """This decorator ensures that any missing inputs in requests are filled with default values specified by the user.

    Default values should be NumPy arrays without batch axis.

    If you plan to group requests ex. with
    [@group_by_keys][pytriton.decorators.group_by_keys] or
    [@group_by_vales][pytriton.decorators.group_by_values] decorators
    provide default values for optional parameters at the beginning of decorators stack.
    The other decorators can then group requests into bigger batches resulting in a better model performance.

    Typical use:
    ```python
    @fill_optionals()
    @group_by_keys()
    @batch
    def infer_fun(**inputs):
        ...
        return outputs
    ```

    Args:
        defaults: keyword arguments containing default values for missing inputs


    If you have default values for some optional parameter it is good idea to provide them at the very beginning,
    so the other decorators (e.g. @group_by_keys) can make bigger consistent groups.
    """

    def _verify_defaults(model_config: TritonModelConfig):
        inputs = {spec.name: spec for spec in model_config.inputs}
        not_matching_default_names = sorted(set(defaults) - set(inputs))
        if not_matching_default_names:
            raise PyTritonBadParameterError(f"Could not found {', '.join(not_matching_default_names)} inputs")

        non_numpy_items = {k: v for k, v in defaults.items() if not isinstance(v, np.ndarray)}
        if non_numpy_items:
            raise PyTritonBadParameterError(
                f"Could not use {', '.join([f'{k}={v}' for k, v in non_numpy_items.items()])} defaults "
                "as they are not NumPy arrays"
            )

        not_matching_dtypes = {k: (v.dtype, inputs[k].dtype) for k, v in defaults.items() if v.dtype != inputs[k].dtype}
        if not_matching_dtypes:
            non_matching_dtypes_str_list = [
                f"{name}: dtype={have_dtype} expected_dtype={expected_dtype}"
                for name, (have_dtype, expected_dtype) in not_matching_dtypes.items()
            ]
            raise PyTritonBadParameterError(
                f"Could not use {', '.join(non_matching_dtypes_str_list)} "
                f"defaults as they have different than input signature dtypes"
            )

        def _shape_match(_have_shape, _expected_shape):
            return len(_have_shape) == len(_expected_shape) and all(
                e == -1 or h == e for h, e in zip(_have_shape, _expected_shape)
            )

        not_matching_shapes = {
            k: (v.shape, inputs[k].shape) for k, v in defaults.items() if not _shape_match(v.shape, inputs[k].shape)
        }
        if not_matching_shapes:
            non_matching_shapes_str_list = [
                f"{name}: shape={have_shape} expected_shape={expected_shape}"
                for name, (have_shape, expected_shape) in not_matching_shapes.items()
            ]
            raise PyTritonBadParameterError(
                f"Could not use {', '.join(non_matching_shapes_str_list)} "
                f"defaults as they have different than input signature shapes"
            )

    @wrapt.decorator
    def _wrapper(wrapped, instance, args, kwargs):
        model_config = get_model_config(wrapped, instance)
        _verify_defaults(model_config)
        # verification if not after group wrappers is in group wrappers

        (requests,) = args

        model_supports_batching = model_config.batching
        for request in requests:
            batch_size = get_inference_request_batch_size(request) if model_supports_batching else None
            for default_key, default_value in defaults.items():
                if default_key in request:
                    continue

                if model_supports_batching:
                    ones_reps = (1,) * default_value.ndim  # repeat once default_value on each axis
                    axis_reps = (batch_size,) + ones_reps  # ... except on batch axis. we repeat it batch_size times
                    default_value = np.tile(default_value, axis_reps)

                request[default_key] = default_value
        return wrapped(*args, **kwargs)

    return _wrapper


@wrapt.decorator
def triton_context(wrapped, instance, args, kwargs):
    """Adds triton context.

    It gives you additional argument passed to the function in **kwargs called 'triton_context'.
    You can read model config from it and in the future possibly have some interaction with triton.
    """
    kwargs[TRITON_CONTEXT_FIELD_NAME] = get_triton_context(wrapped, instance)
    return wrapped(*args, **kwargs)


@wrapt.decorator
def pad_batch(wrapped, instance, args, kwargs):
    """Add padding to the inputs batches.

    Decorator appends last rows to the inputs multiple times to get desired batch size (preferred batch size or
    max batch size from model config whatever is closer to current input size).
    """
    inputs = {k: v for k, v in kwargs.items() if k != "__triton_context__"}
    first_input = next(iter(inputs.values()))
    config = get_model_config(wrapped, instance)
    batch_sizes = (
        []
        if (config.batcher is None or config.batcher.preferred_batch_size is None)
        else sorted(config.batcher.preferred_batch_size)
    )
    batch_sizes.append(config.max_batch_size)
    batch_size = batch_sizes[bisect_left(batch_sizes, first_input.shape[0])]

    new_inputs = {
        input_name: np.repeat(
            input_array,
            np.concatenate(
                [np.ones(input_array.shape[0] - 1), np.array([batch_size - input_array.shape[0] + 1])]
            ).astype(np.int64),
            axis=0,
        )
        for input_name, input_array in inputs.items()
    }

    kwargs.update(new_inputs)
    return wrapped(*args, **kwargs)


_SPECIAL_KEYS = ["__triton_context__"]


def first_value(*keys: str, squeeze_single_values=True, strict: bool = True):
    """This decorator overwrites selected inputs with first element of the given input.

    It can be used in two ways:

    1. Wrapping a single request inference callable by chaining with @batch decorator:
        ```python
        @batch
        @first_value("temperature")
        def infer_fn(**inputs):
            ...
            return result
        ```

    2. Wrapping a multiple requests inference callable:
        ```python
        @first_value("temperature")
        def infer_fn(requests):
            ...
            return results
        ```

    By default, the decorator squeezes single value arrays to scalars.
    This behavior can be disabled by setting the `squeeze_single_values` flag to False.

    By default, the decorator checks the equality of the values on selected values.
    This behavior can be disabled by setting the `strict` flag to False.

    Wrapper can only be used with models that support batching.

    Args:
        keys: The input keys selected for conversion.
        squeeze_single_values: squeeze single value ND array to scalar values. Defaults to True.
        strict: enable checking if all values on single selected input of request are equal. Defaults to True.

    Raises:
        PyTritonRuntimeError: if not all values on a single selected input of the request are equal
        and the strict flag is set to True. Additionally, if the decorator is used with a model that doesn't support batching,
        PyTritonBadParameterError: if any of the keys passed to the decorator are not allowed.
    """
    if any(k in _SPECIAL_KEYS for k in keys):
        not_allowed_keys = [key for key in keys if key in _SPECIAL_KEYS]
        raise PyTritonBadParameterError(
            f"The keys {', '.join(not_allowed_keys)} are not allowed as keys for @first_value wrapper. "
            f"The set of not allowed keys are {', '.join(_SPECIAL_KEYS)}"
        )

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        model_config = get_model_config(wrapped, instance)
        if not model_config.batching:
            raise PyTritonRuntimeError("The @first_value decorator can only be used with models that support batching.")

        def _replace_inputs_with_first_value(_request):
            for input_name in keys:
                if input_name not in _request:
                    continue

                values = _request[input_name]
                if strict:
                    # do not set axis for arrays with strings (object) or models not supporting batching
                    axis_of_uniqueness = None if values.dtype == object else 0
                    unique_values = np.unique(values, axis=axis_of_uniqueness)
                    if len(unique_values) > 1:
                        raise PyTritonRuntimeError(
                            f"The values on the {input_name!r} input are not equal. "
                            "To proceed, either disable strict mode in @first_value wrapper "
                            "or ensure that the values always are consistent. "
                            f"The current values of {input_name!r} are {_request[input_name]!r}."
                        )

                _first_value = values[0]
                if (
                    squeeze_single_values
                    and not np.isscalar(_first_value)
                    and all(dim == 1 for dim in _first_value.shape)
                ):
                    _dim_0_array = np.squeeze(_first_value)
                    _first_value = _dim_0_array[()]  # obtain scalar from 0-dim array with numpy type

                _request[input_name] = _first_value
            return _request

        inputs_names = set(kwargs) - set(_SPECIAL_KEYS)
        if inputs_names:
            kwargs = _replace_inputs_with_first_value(kwargs)
            return wrapped(*args, **kwargs)
        else:
            requests, *other_args = args
            requests = [_replace_inputs_with_first_value(request) for request in requests]
            return wrapped(requests, *other_args, **kwargs)

    return wrapper
