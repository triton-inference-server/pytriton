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

import pathlib
import unittest.mock

import numpy as np
import psutil
import pytest

from pytriton.proxy.communication import TensorStore, _DataBlocksServer, serialize_numpy_with_struct_header


@pytest.fixture(scope="function")
def tensor_store(tmp_path):
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)
    try:
        tensor_store.start()
        yield tensor_store
    finally:
        tensor_store.close()


def test_tensor_store_create_and_close(tmp_path):
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)  # authkey will be taken from current process
    tensor_store.start()

    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        assert len(children) == 2  # block store + resource tracker side processes should be created
    finally:
        tensor_store.close()

    children = current_process.children(recursive=True)
    assert len(children) == 1  # block store side process should be closed; resource_tracker should be still running


def test_tensor_store_unregisters_shm_from_resource_tracker(tmp_path, mocker):
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)  # authkey will be taken from current process
    tensor_store.start()  # start new block store side process

    a = np.zeros((10, 10), dtype=np.float32)
    b = np.array([b"foo", b"longer_bar"])
    c = np.array([b"foo", b"longer_bar"], dtype=object)

    from multiprocessing import resource_tracker

    spy_unregister = mocker.spy(resource_tracker, "unregister")

    tensor_store.put([a, b, c])

    shm_names = {
        shm._name for shm, tensor_ref in tensor_store._handled_blocks.values()  # pytype: disable=attribute-error
    }

    tensor_store.close()

    expected_unregister_calls = [unittest.mock.call(shm_name, "shared_memory") for shm_name in shm_names]
    spy_unregister.assert_has_calls(expected_unregister_calls, any_order=True)


def test_tensor_store_shared_memory_unlinked_on_tensor_store_close(tmp_path):
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)  # authkey will be taken from current process
    tensor_store.start()  # start new block store side process

    a = np.zeros((10, 10), dtype=np.float32)
    b = np.array([b"foo", b"longer_bar"])
    c = np.array([b"foo", b"longer_bar"], dtype=object)

    tensor_store.put([a, b, c])

    shm_names = {
        shm._name for shm, tensor_ref in tensor_store._handled_blocks.values()  # pytype: disable=attribute-error
    }

    for shm_name in shm_names:
        shm_path = pathlib.Path("/dev/shm") / shm_name[1:]
        assert shm_path.exists()  # shared memory should be present

    tensor_store.close()

    for shm_name in shm_names:
        shm_path = pathlib.Path("/dev/shm") / shm_name[1:]
        assert not shm_path.exists()  # shared memory should be unlinked


def test_tensor_store_connection_timeout(tmp_path):
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)

    with pytest.raises(TimeoutError):
        tensor_store.connect(timeout_s=0.05)

    tensor_store.start()
    tensor_store.connect(timeout_s=0.05)


# 12bytes is 3x4bytes (num of segments and 2 segments sizes - header + np array)
_flat_array_header_size = len(serialize_numpy_with_struct_header(np.zeros((1,), dtype=np.int8))[0]) + 12


@pytest.mark.parametrize(
    "tensors, n_times",
    (
        # different dtypes
        (
            [
                np.zeros((10, 10), dtype=np.float32),
                np.array([b"foo", b"longer_bar"]),
                np.array([b"foo", b"longer_bar"], dtype=object),
            ],
            1,
        ),
        # case when tensors are larger than minimal segment size
        (
            [
                np.zeros(
                    (_DataBlocksServer._minimal_segment_size // np.dtype(np.float32).itemsize * 3,), dtype=np.float32
                ),
                np.zeros(
                    (_DataBlocksServer._minimal_segment_size // np.dtype(np.float32).itemsize * 5), dtype=np.float32
                ),
                np.zeros(
                    int(_DataBlocksServer._minimal_segment_size // np.dtype(np.float32).itemsize * 0.75),
                    dtype=np.float32,
                ),
            ],
            1,
        ),
        # size match exactly single segment, thus free_blocks should be empty
        (
            [
                np.zeros(
                    (
                        (_DataBlocksServer._minimal_segment_size - _flat_array_header_size)
                        // np.dtype(np.int8).itemsize,
                    ),
                    dtype=np.int8,
                ),
            ],
            2,
        ),
    ),
)
def test_tensor_store_get_put_equal(tensor_store, tensors, n_times):
    for _ in range(n_times):
        tensors_ids = tensor_store.put(tensors)
        assert len(tensors) == len(tensors_ids)
        for tensor, tensor_id in zip(tensors, tensors_ids):
            tensor_retrieved = tensor_store.get(tensor_id)
            np.testing.assert_equal(tensor, tensor_retrieved)
