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
"""Communication utility module.

It is used for interaction between model and proxy_backend.
"""

import dataclasses
import json
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import numpy as np


# copy from
# https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py
def _serialize_byte_tensor(input_tensor):
    """Serializes a bytes tensor into a flat numpy array of length prepended bytes.

    The numpy array should use dtype of np.object_. For np.bytes_,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.

    Args:
        input_tensor: The bytes tensor to serialize.

    Returns:
    serialized array as bytes buffer.

    Raises:
        UnicodeEncodeErrors: raised when try to cast to string of non-bytes items fails
    """
    if input_tensor.size == 0:
        return ()

    # If the input is a tensor of string/bytes objects, then must flatten those
    # into a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C" order.
    if (input_tensor.dtype == np.object_) or (input_tensor.dtype.type == np.bytes_):
        flattened_ls = []
        total_len = 0
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order="C"):
            # If directly passing bytes to BYTES type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if input_tensor.dtype == np.object_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = str(obj.item()).encode("utf-8")
            else:
                s = obj.item()
            item_len = len(s)
            flattened_ls.append(struct.pack("<I", item_len))
            flattened_ls.append(s)
            total_len += 4 + item_len
        flattened_ls.insert(0, struct.pack("<I", total_len))
        flattened = b"".join(flattened_ls)
        return flattened
    return None


# copy from
# https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py
def _deserialize_bytes_tensor(encoded_tensor, dtype):
    """Deserializes an encoded bytes tensor into an numpy array of dtype of python objects.

    Args:
        encoded_tensor : The encoded bytes tensor where each element has its length in
        first 4 bytes followed by the content

    Returns:
    The 1-D numpy array of type object containing the deserialized bytes in 'C' order.
    """
    strs = []
    offset = 0
    val_buf = encoded_tensor
    val_len = struct.unpack_from("<I", val_buf, offset)[0] + 4
    offset += 4
    while offset < val_len:
        item_length = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        item = struct.unpack_from(f"<{item_length}s", val_buf, offset)[0]
        offset += item_length
        strs.append(item)
    return np.array(strs, dtype=dtype)


@dataclasses.dataclass
class TensorInfo:
    """Data class for storing numpy array schema and buffer information."""

    memory_range: Tuple[int, int]
    shape: Tuple[int, ...]
    dtype: str


@dataclasses.dataclass
class Request:
    """Object transferred from proxy backend to callback handler containing input data."""

    inputs: List[Dict[str, TensorInfo]]
    memory_name: str

    @classmethod
    def from_bytes(cls, content: bytes) -> "Request":
        """Reconstruct Request object from bytes.

        Args:
            content: bytes to parse
        """
        request_dict = json.loads(content)
        request_list = request_dict["inputs"]

        return cls(
            inputs=[
                {input_name: TensorInfo(**tensor_info) for input_name, tensor_info in req_dict.items()}
                for req_dict in request_list
            ],
            memory_name=request_dict["memory_name"],
        )

    def as_bytes(self) -> bytes:
        """Serializes Request object to bytes."""
        request_dict = {
            "inputs": [
                {input_name: dataclasses.asdict(tensor_info) for input_name, tensor_info in req_dict.items()}
                for req_dict in self.inputs
            ],
            "memory_name": self.memory_name,
        }
        return json.dumps(request_dict).encode("utf-8")


@dataclasses.dataclass
class Response:
    """Object transferred from callback handler containing output data."""

    outputs: Optional[List[Dict[str, TensorInfo]]] = None
    error: Optional[str] = None
    memory_name: Optional[str] = None

    @classmethod
    def from_bytes(cls, content: bytes) -> "Response":
        """Reconstruct Response object from bytes.

        Args:
            content: bytes to parse
        """
        response_dict = json.loads(content)
        outputs = response_dict.get("outputs", [])
        memory_name = response_dict.get("memory_name", None)
        return cls(
            outputs=[
                {output_name: TensorInfo(**tensor_info) for output_name, tensor_info in resp_dict.items()}
                for resp_dict in outputs
            ],
            error=response_dict.get("error"),
            memory_name=memory_name,
        )

    def as_bytes(self) -> bytes:
        """Serializes Response object to bytes."""
        result = {"error": self.error, "memory_name": self.memory_name}
        if self.outputs:
            result["outputs"] = [
                {output_name: dataclasses.asdict(tensor_info) for output_name, tensor_info in resp_dict.items()}
                for resp_dict in self.outputs
            ]
        return json.dumps(result).encode("utf-8")


class ShmManager:
    """Controls transfer between input and output numpy arrays in via shared memory."""

    def __init__(self):
        """Initialize ShmManager class."""
        self._shm_buffer = None
        self._serialized_arrays = {}
        self._memory_index = 0

    def _init_buffer(self, _required_buffer_size: int):
        if self._shm_buffer and self._shm_buffer.size < _required_buffer_size:
            self.dispose()

        if self._shm_buffer is None:
            self._shm_buffer = SharedMemory(create=True, size=_required_buffer_size)

        self._memory_index = 0

    def _calc_required_buffer(self, array_dicts: List[Dict[str, np.ndarray]]) -> int:
        self._serialized_arrays.clear()
        required_buffer_size_sum = 0
        for index, input_dict in enumerate(array_dicts):
            for input_name, np_array in input_dict.items():
                if np_array.dtype == np.object_ or np_array.dtype.type == np.bytes_:
                    serialized_np_array = _serialize_byte_tensor(np_array)
                    required_buffer_size = len(serialized_np_array)
                    self._serialized_arrays[(index, input_name)] = serialized_np_array
                else:
                    required_buffer_size = np_array.nbytes
                required_buffer_size_sum += required_buffer_size
        return required_buffer_size_sum

    def _get_buffer_for_write(self, size: int) -> Tuple[memoryview, Tuple[int, int]]:
        buf_range = (self._memory_index, self._memory_index + size)
        sub_buf = self._shm_buffer.buf[buf_range[0] : buf_range[1]]
        self._memory_index += size
        return sub_buf, buf_range

    def _get_buffer_for_read(self, buf_range: Tuple[int, int]) -> memoryview:
        sub_buf = self._shm_buffer.buf[buf_range[0] : buf_range[1]]
        return sub_buf

    def memory_name(self) -> Optional[str]:
        """Returns name of shared memory buffer."""
        return self._shm_buffer.name if self._shm_buffer else None

    def to_shm(self, array_dicts: List[Dict[str, np.ndarray]]) -> List[Dict[str, TensorInfo]]:
        """Serialize list of requests or responses to shared memory.

        Args:
            array_dicts: input list of request dicts for serialization
        Returns:
            coresponding structure where each numpy array is replaced with TensorInfo description
        """
        self._serialized_arrays.clear()
        required_buffer_size_sum = self._calc_required_buffer(array_dicts)
        self._init_buffer(required_buffer_size_sum)

        return [
            {input_name: self._wrap_array(req_index, input_name, input_dict[input_name]) for input_name in input_dict}
            for req_index, input_dict in enumerate(array_dicts)
        ]

    def from_shm(self, infos_dict_list: List[Dict[str, TensorInfo]], memory_name: str) -> List[Dict[str, np.ndarray]]:
        """Deserialize list of requests or responses from shared memory.

        Args:
            infos_dict_list: input list of request dicts of TensorInfo for deserialization
            memory_name: share memory buffer name

        Returns:
            list of request dicts deserialized from shared memory
        """
        if self._shm_buffer and self._shm_buffer.name != memory_name:
            self.dispose()
        if self._shm_buffer is None:
            self._shm_buffer = SharedMemory(memory_name, create=False)

        results = [
            {input_name: self.as_np_array(info) for input_name, info in req_dict.items()}
            for req_dict in infos_dict_list
        ]
        return results

    def _wrap_array(self, req_index: int, input_name: str, np_array: np.ndarray) -> TensorInfo:
        """Copies numpy array data to shared memory.

        Reuse shared memory if possible. Reallocates shared memory if needed.

        Args:
            req_index: request index
            input_name: input name
            np_array: source numpy array
        """
        if np_array.dtype == np.object_ or np_array.dtype.type == np.bytes_:
            serialized_np_array = self._serialized_arrays[(req_index, input_name)]  # _serialize_byte_tensor(np_array)
            required_buffer_size = len(serialized_np_array)
            buf, buf_range = self._get_buffer_for_write(required_buffer_size)
            buf[:required_buffer_size] = serialized_np_array
        else:
            required_buffer_size = np_array.nbytes
            buf, buf_range = self._get_buffer_for_write(required_buffer_size)
            shm_array = np.ndarray(shape=np_array.shape, dtype=np_array.dtype, buffer=buf)
            shm_array[:] = np_array
        return TensorInfo(buf_range, np_array.shape, str(np_array.dtype))

    def dispose(self):
        """Free resources used by this wrapper."""
        if self._shm_buffer:
            self._shm_buffer.close()
            try:
                self._shm_buffer.unlink()
            except FileNotFoundError:
                from multiprocessing.resource_tracker import unregister

                # to WAR bug in SharedMemory:
                # not unregistering shared memory segment in current process resource tracker
                # when /dev/shm file doesn't exist - removed by other process
                unregister(self._shm_buffer._name, "shared_memory")  # pytype: disable=attribute-error
        self._shm_buffer = None

    def as_np_array(self, info: TensorInfo) -> np.ndarray:
        """Create numpy array based on shared memory buffer.

        Args:
            info Tensor info for deserialization

        Returns:
            numpy array based on shared memory.
        """
        dtype = np.dtype(info.dtype)
        buf = self._get_buffer_for_read(info.memory_range)

        if dtype == np.object_ or dtype.type == np.bytes_:
            return _deserialize_bytes_tensor(buf, dtype).reshape(info.shape)
        else:
            return np.ndarray(shape=info.shape, dtype=dtype, buffer=buf)
