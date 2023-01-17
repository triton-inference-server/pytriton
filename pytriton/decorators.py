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
"""Inference function decorators."""
import dataclasses
import itertools
import operator
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import wrapt

from pytriton.constants import TRITON_CONTEXT_FIELD_NAME
from pytriton.exceptions import PytritonBadParameterError, PytritonValidationError
from pytriton.model_config.triton_model_config import TritonModelConfig


def _get_triton_request_batch_size(triton_request):
    first_input_value = next(iter(triton_request.values()))
    batch_size, *dims = first_input_value.shape
    return batch_size


@dataclasses.dataclass
class TritonContext:
    """Triton context definition class."""

    model_config: TritonModelConfig


def get_triton_context(wrapped, instance) -> TritonContext:
    """Retrieves triton context from callable.

    It is used in @triton_context to get triton context registered by triton binding in inference callable.
    If you use @triton_context decorator you do not need this function.
    """
    caller = instance or wrapped
    if not hasattr(caller, "__triton_context__"):
        raise PytritonValidationError("Wrapped function or object must bound with triton to get  __triton_context__")
    return caller.__triton_context__


def get_model_config(wrapped, instance) -> TritonModelConfig:
    """Retrieves triton model config from callable.

    It is internally used in convert_output function to get output list from model.
    You can use this in custom decorators if you need access to model_config information.
    If you use @triton_context decorator you do not need this function (you can get model_config from triton_context).
    """
    return get_triton_context(wrapped, instance).model_config


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
            raise PytritonValidationError("Outputs length different than config outputs length")
        outputs = {config_output.name: output for config_output, output in zip(model_config.outputs, outputs)}
        return outputs
    else:
        raise PytritonValidationError(f"Unsupported output type {type(outputs)}.")


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
    It passes **kwargs to inference function where each named input contains numpy array with batch of requests
    received by Triton server.
    We assume that each request has the same set of keys (you can use group_by_keys decorator before
    using @batch decorator if your requests may have different set of keys).
    """
    req_list = args[0]
    input_names = req_list[0].keys()

    for req_dict2 in req_list[1:]:
        if input_names != req_dict2.keys():
            raise PytritonValidationError("Cannot batch requests with different set of inputs keys")

    inputs = {}
    for model_input in input_names:
        concatenated_input_data = np.concatenate([req_dict[model_input] for req_dict in req_list])
        inputs[model_input] = concatenated_input_data

    args = args[1:]
    new_kwargs = dict(kwargs)
    new_kwargs.update(inputs)
    outputs = wrapped(*args, **new_kwargs)

    outputs = convert_output(outputs, wrapped, instance)
    output_names = outputs.keys()

    out_list = []
    start_idx = 0
    for request in req_list:
        # get batch_size of first input for each request - assume that all inputs have same batch_size
        first_input = next(iter(request.values()))
        request_batch_size = first_input.shape[0]
        req_output_dict = {}
        for _output_ind, output_name in enumerate(output_names):
            req_output = outputs[output_name][start_idx : start_idx + request_batch_size, ...]
            req_output_dict[output_name] = req_output
        out_list.append(req_output_dict)
        start_idx += request_batch_size
    return out_list


def group_by_values(*keys):
    """Group by values.

    Decorator prepares groups of requests with the same input value (for selected keys) and calls wrapped function
    for each group separately (it is especially convenient to use with models that requires dynamic parameters
    sent by the user e.g. temperature - in this case we would like to run model only for requests with the same
    temperature value)
    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        """Group by values."""
        inputs = args[0]

        idx_inputs = []
        for idx, input in enumerate(inputs):
            req_key_list = []
            for requested_key in keys:
                if requested_key in input:
                    arr = np.unique(input[requested_key], axis=0)
                    key = (arr.shape, tuple(arr.flatten()))
                    req_key_list.append(key)
                else:
                    req_key_list.append(((-1,), (-1,)))
            req_key = tuple(req_key_list)
            idx_inputs.append((idx, req_key, input))

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

    return wrapper


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

        @fill_optionals()
        @group_by_keys()
        @batch
        def infer_fun(**inputs):
            ...
            return outputs

    Args:
        defaults: keyword arguments containing default values for missing inputs


    If you have default values for some optional parameter it is good idea to provide them at the very beginning,
    so the other decorators (e.g. @group_by_keys) can make bigger consistent groups.
    """

    def _verify_defaults(_triton_context: TritonContext):
        inputs = {spec.name: spec for spec in _triton_context.model_config.inputs}
        not_matching_default_names = sorted(set(defaults) - set(inputs))
        if not_matching_default_names:
            raise PytritonBadParameterError(f"Could not found {', '.join(not_matching_default_names)} inputs")

        non_numpy_items = {k: v for k, v in defaults.items() if not isinstance(v, np.ndarray)}
        if non_numpy_items:
            raise PytritonBadParameterError(
                f"Could not use {', '.join([f'{k}={v}' for k, v in non_numpy_items.items()])} defaults "
                "as they are not NumPy arrays"
            )

        not_matching_dtypes = {k: (v.dtype, inputs[k].dtype) for k, v in defaults.items() if v.dtype != inputs[k].dtype}
        if not_matching_dtypes:
            non_matching_dtypes_str_list = [
                f"{name}: dtype={have_dtype} expected_dtype={expected_dtype}"
                for name, (have_dtype, expected_dtype) in not_matching_dtypes.items()
            ]
            raise PytritonBadParameterError(
                f"Could not use {', '.join(non_matching_dtypes_str_list)} "
                f"defaults as they have different than input signature dtypes"
            )

        def _shape_match(_have_shape, _expected_shape):
            return len(_have_shape) == len(_expected_shape) and all(
                [e == -1 or h == e for h, e in zip(_have_shape, _expected_shape)]
            )

        not_matching_shapes = {
            k: (v.shape, inputs[k].shape) for k, v in defaults.items() if not _shape_match(v.shape, inputs[k].shape)
        }
        if not_matching_shapes:
            non_matching_shapes_str_list = [
                f"{name}: shape={have_shape} expected_shape={expected_shape}"
                for name, (have_shape, expected_shape) in not_matching_shapes.items()
            ]
            raise PytritonBadParameterError(
                f"Could not use {', '.join(non_matching_shapes_str_list)} "
                f"defaults as they have different than input signature shapes"
            )

    @wrapt.decorator
    def _wrapper(wrapped, instance, args, kwargs):
        _triton_context = get_triton_context(wrapped, instance)

        _verify_defaults(_triton_context)
        # verification if not after group wrappers is in group wrappers

        (requests,) = args

        model_supports_batching = _triton_context.model_config.batching
        for request in requests:
            batch_size = _get_triton_request_batch_size(request) if model_supports_batching else None
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


def first_value(*keys):
    """This decorator takes first element from batch.

    If the value is one element array, it is converted to scalar value.
    It is useful for dynamic arguments sent in requests.

    Args:
        keys: input keys selected for conversion
    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        inputs = {k: v for k, v in kwargs.items() if k != "__triton_context__"}
        for key in keys:
            res = inputs[key][0]
            if max(res.shape) == 1:
                res = res.flatten().item()
            inputs[key] = res
        kwargs.update(inputs)
        return wrapped(*args, **kwargs)

    return wrapper
