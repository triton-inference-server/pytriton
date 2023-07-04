# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np
import pytest

from pytriton.proxy.communication import (
    calc_serialized_size_of_numpy_with_struct_header,
    deserialize_numpy_with_struct_header,
    serialize_numpy_with_struct_header,
)

# subset of test cases from https://github.com/dask/distributed/blob/main/distributed/protocol/tests/test_numpy.py
_test_cases = [
    np.ones(5),
    np.array(5),
    np.random.random((5, 5)),
    np.random.random((5, 5))[::2, :],
    np.random.random((5, 5))[:, ::2],
    np.asfortranarray(np.random.random((5, 5))),
    np.asfortranarray(np.random.random((5, 5)))[::2, :],
    np.asfortranarray(np.random.random((5, 5)))[:, ::2],
    np.random.random(5).astype("f4"),
    np.random.random(5).astype(">i8"),
    np.random.random(5).astype("<i8"),
    np.array([True, False, True]),
    np.array(["abc"], dtype="S3"),
    np.array(["abc"], dtype="U3"),
    np.array([[b"hello\x00\x00\x00\x00", b"world"], [b"foo", b"bar\x00\x00"]], dtype=object),
    np.zeros(5000, dtype="S32"),
    np.zeros((1, 1000, 1000)),
    np.arange(12)[::2],  # non-contiguous array
    np.broadcast_to(np.arange(3), shape=(10, 3)),  # zero-strided array
    np.array(
        [
            [
                b"<|endoftext|>",
                b"1",
                b" 2",
                b" 3",
                b" 4",
                b" 5",
                b" 6",
                b" 7",
                b" 8",
                b" 9",
                b" 10",
                b" 11",
                b" 12",
                b" 13",
                b" 14",
                b" 15",
                b" 16",
                b" 17",
                b" 18",
                b" 19",
                b" 20",
                b" 21",
                b" 22",
                b" 23",
                b" 24",
                b" 25",
                b" 26",
                b" 27",
                b" 28",
                b" 29",
                b" 30",
                b" 31",
                b" 32",
                b" 33",
            ]
        ]
    ),
]


@pytest.mark.parametrize("x", _test_cases)
def test_serialize_and_deserialize_np_array(x):
    frames = serialize_numpy_with_struct_header(x)
    assert all(isinstance(frame, (bytes, memoryview)) for frame in frames)
    y = deserialize_numpy_with_struct_header(frames)
    assert x.shape == y.shape, (x.shape, y.shape)
    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    if x.flags.c_contiguous or x.flags.f_contiguous:
        assert x.strides == y.strides, (x.strides, y.strides)
    np.testing.assert_equal(x, y)


@pytest.mark.parametrize("x", _test_cases)
def test_calc_serialized_size_of_numeric_np_array(x):
    frames = serialize_numpy_with_struct_header(x)
    size = calc_serialized_size_of_numpy_with_struct_header(x)
    assert size == [memoryview(frame).nbytes for frame in frames]
