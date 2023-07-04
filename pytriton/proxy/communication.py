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
"""Communication utility module.

It is used for interaction between model and proxy_backend.
"""

import dataclasses
import json
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

# copy from
# https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py


def _serialize_byte_tensor(tensor) -> bytes:
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
    if tensor.size == 0:
        return b""

    # If the input is a tensor of string/bytes objects, then must flatten those
    # into a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C" order.
    assert (tensor.dtype == np.object_) or (tensor.dtype.type == np.bytes_)
    flattened_ls = []
    total_len = 0
    for obj in np.nditer(tensor, flags=["refs_ok"], order="C"):
        # If directly passing bytes to BYTES type,
        # don't convert it to str as Python will encode the
        # bytes which may distort the meaning
        if tensor.dtype == np.object_ and type(obj.item()) != bytes:
            s = str(obj.item()).encode("utf-8")
        else:
            s = obj.item()
        item_len = len(s)
        flattened_ls.append(struct.pack("<I", item_len))
        flattened_ls.append(s)
        total_len += struct.calcsize("<I") + item_len
    flattened_ls.insert(0, struct.pack("<I", total_len))
    flattened = b"".join(flattened_ls)
    return flattened


# copy from
# https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py
def _deserialize_bytes_tensor(encoded_tensor, dtype, order: Literal["C", "F"] = "C") -> np.ndarray:
    """Deserializes an encoded bytes tensor into an numpy array of dtype of python objects.

    Args:
        encoded_tensor : The encoded bytes tensor where each element has its length in
        first 4 bytes followed by the content
        dtype: The dtype of the numpy array to deserialize to.
        order: The order of the numpy array to deserialize to.

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
    return np.array(strs, dtype=dtype, order=order)


_MAX_DTYPE_DESCR = 8
_PARTIAL_HEADER_FORMAT = f"<{_MAX_DTYPE_DESCR}scH"


def _pack_header(shape: Tuple[int, ...], dtype: np.dtype, order: Literal["C", "F"] = "C") -> bytes:
    header_format = _PARTIAL_HEADER_FORMAT + "Q" * len(shape)
    dtype_descr = np.lib.format.dtype_to_descr(dtype)
    assert (
        len(dtype_descr) <= _MAX_DTYPE_DESCR
    ), f"dtype descr is too long; dtype_descr={dtype_descr} max={_MAX_DTYPE_DESCR}"
    return struct.pack(header_format, dtype_descr.encode("utf-8"), order.encode("ascii"), len(shape), *shape)


def _unpack_header(header: bytes) -> Tuple[Tuple[int, ...], np.dtype, Literal["C", "F"]]:
    shape_offset = struct.calcsize(_PARTIAL_HEADER_FORMAT)
    dtype_descr, order, ndim = struct.unpack_from(_PARTIAL_HEADER_FORMAT, header, offset=0)
    shape = struct.unpack_from("Q" * ndim, header, offset=shape_offset)
    dtype = np.lib.format.descr_to_dtype(dtype_descr.decode("utf-8").rstrip("\x00"))
    order = order.decode("ascii")
    return shape, dtype, order


def serialize_numpy_with_struct_header(tensor: np.ndarray) -> List[Union[bytes, memoryview]]:
    """Serialize numpy array to list of bytes and memoryviews.

    Args:
        tensor: numpy array to serialize

    Returns:
        List of data frames in form of bytes and memoryviews
    """
    if tensor.dtype.hasobject:
        data = _serialize_byte_tensor(tensor.ravel())
        order = "C"  # as _serialize_byte_tensor returns C-ordered array
    else:
        if not tensor.data.contiguous:
            tensor = np.ascontiguousarray(tensor)
        data = tensor.data
        order = "C" if tensor.flags.c_contiguous else "F"

    header = _pack_header(tensor.shape, tensor.dtype, order)
    frames = [header, data]
    return frames


def deserialize_numpy_with_struct_header(frames: List[Union[bytes, memoryview]]) -> np.ndarray:
    """Deserialize numpy array from list of bytes and memoryviews.

    Args:
        frames: List of data frames in form of bytes and memoryviews

    Returns:
        numpy array
    """
    header, data = frames
    shape, dtype, order = _unpack_header(header)
    if dtype.hasobject:
        tensor = _deserialize_bytes_tensor(data, dtype).reshape(shape)
    else:
        tensor = np.ndarray(shape, dtype=dtype, buffer=data, order=order)
    return tensor


def calc_serialized_size_of_numpy_with_struct_header(tensor: np.ndarray) -> List[int]:
    """Calculate size of serialized numpy array.

    Args:
        tensor: numpy array to serialize

    Returns:
        List of sizes of data frames
    """
    header_size = struct.calcsize(_PARTIAL_HEADER_FORMAT) + struct.calcsize("Q") * len(tensor.shape)
    if tensor.dtype.hasobject:
        items_sizes = []
        order = "C" if tensor.flags.c_contiguous else "F"
        for obj in np.nditer(tensor, flags=["refs_ok"], order=order):
            if tensor.dtype == np.object_ and type(obj.item()) != bytes:
                s = str(obj.item()).encode("utf-8")
            else:
                s = obj.item()
            items_sizes.append(len(s))

        # total_size + for size of each item + each item
        data_size = struct.calcsize("<I") + struct.calcsize("<I") * len(items_sizes) + sum(items_sizes)
    else:
        data_size = tensor.nbytes

    return [header_size, data_size]


@dataclasses.dataclass(frozen=True)
class TensorId:
    """Data class for storing id of tensor in Tensor Store."""

    memory_name: str
    memory_offset: int

    @classmethod
    def from_str(cls, tensor_id: str):
        """Create TensorId object from string."""
        memory_name, memory_offset = tensor_id.split(":", maxsplit=1)
        return cls(memory_name, int(memory_offset))

    def __str__(self):
        """Convert TensorId object to string."""
        return f"{self.memory_name}:{self.memory_offset}"


@dataclasses.dataclass
class MetaRequestResponse:
    """Data class for storing input/output data and parameters."""

    data: Dict[str, TensorId]
    parameters: Optional[Dict[str, Union[str, int, bool]]] = None


@dataclasses.dataclass
class InferenceHandlerRequests:
    """Object transferred from proxy backend to callback handler containing input data."""

    requests: List[MetaRequestResponse]

    @classmethod
    def from_bytes(cls, content: bytes) -> "InferenceHandlerRequests":
        """Reconstruct InferenceHandlerRequests object from bytes.

        Args:
            content: bytes to parse
        """
        requests = json.loads(content)
        return cls(
            requests=[
                MetaRequestResponse(
                    data={
                        input_name: TensorId.from_str(tensor_id)
                        for input_name, tensor_id in request.get("data", {}).items()
                    },
                    parameters=request.get("parameters"),
                )
                for request in requests["requests"]
            ]
        )

    def as_bytes(self) -> bytes:
        """Serializes InferenceHandlerRequests object to bytes."""
        requests = {
            "requests": [
                {
                    "data": {input_name: str(tensor_id) for input_name, tensor_id in request.data.items()},
                    "parameters": request.parameters,
                }
                for request in self.requests
            ]
        }
        return json.dumps(requests).encode("utf-8")


@dataclasses.dataclass
class InferenceHandlerResponses:
    """Object transferred from callback handler containing output data."""

    responses: Optional[List[MetaRequestResponse]] = None
    error: Optional[str] = None

    @classmethod
    def from_bytes(cls, content: bytes) -> "InferenceHandlerResponses":
        """Reconstruct InferenceHandlerResponses object from bytes.

        Args:
            content: bytes to parse
        """
        responses = json.loads(content)
        return cls(
            responses=[
                MetaRequestResponse(
                    {
                        output_name: TensorId.from_str(tensor_id)
                        for output_name, tensor_id in response.get("data", {}).items()
                    }
                )
                for response in responses.get("responses", [])
            ],
            error=responses.get("error"),
        )

    def as_bytes(self) -> bytes:
        """Serializes InferenceHandlerResponses object to bytes."""
        result = {"error": self.error}
        if self.responses:
            result["responses"] = [
                {"data": {output_name: str(tensor_id) for output_name, tensor_id in response.data.items()}}
                for response in self.responses
            ]
        return json.dumps(result).encode("utf-8")


class ShmManager:
    """Controls transfer between input and output numpy arrays in via shared memory."""

    def __init__(self):
        """Initialize ShmManager class."""
        self._shm_buffer = None
        self._serialized_arrays = {}
        self._offset = 0
        self.serialize = serialize_numpy_with_struct_header
        self.deserialize = deserialize_numpy_with_struct_header
        self._calc_serialized_size = calc_serialized_size_of_numpy_with_struct_header

    def calc_serialized_size(self, tensor) -> int:
        """Calculate size of serialized tensor.

        Include frames storage (its sizes and total size).

        Args:
            tensor: numpy array to serialize

        Returns:
            size of serialized tensor
        """
        # frames sizes + total size + frames sizes
        return sum(self._calc_serialized_size(tensor)) + struct.calcsize("<I") + 2 * struct.calcsize("<I")

    def reset_buffer(self, required_buffer_size: int):
        """Reset shared memory buffer.

        Reallocate buffer if it is not big enough.

        Args:
            required_buffer_size: size of buffer to allocate
        """
        if self._shm_buffer and self._shm_buffer.size < required_buffer_size:
            self.dispose()

        if self._shm_buffer is None:
            self._shm_buffer = SharedMemory(create=True, size=required_buffer_size)

        self._offset = 0

    def append(self, tensor: np.ndarray) -> TensorId:
        """Append tensor to shared memory buffer.

        Args:
            tensor: numpy array to append

        Returns:
            TensorId object
        """
        if self._shm_buffer is None:
            raise RuntimeError(
                "Shared memory buffer is not initialized. Call reset_buffer(required_buffer_size) first."
            )

        frames = self.serialize(tensor)
        offset = self._offset
        data_copied_len = self._copy_frames(frames, offset)
        self._offset += data_copied_len
        tensor_id = TensorId(self._shm_buffer.name, offset)
        return tensor_id

    def get(self, tensor_id: Union[str, TensorId]) -> np.ndarray:
        """Get tensor from shared memory buffer.

        Args:
            tensor_id: TensorId object or string representation of TensorId

        Returns:
            numpy array
        """
        if isinstance(tensor_id, str):
            tensor_id = TensorId.from_str(tensor_id)
        frames = self._handle_frames(tensor_id.memory_name, tensor_id.memory_offset)
        tensor = self.deserialize(frames)
        return tensor

    def _copy_frames(self, frames: List[Union[bytes, memoryview]], offset) -> int:
        # caller should ensure that self._shm_buffer is initialized
        shm_buffer: SharedMemory = self._shm_buffer  # type: ignore

        total_size = struct.calcsize("<I")  # start after total_size; max 4GB for all frames
        for frame in frames:
            if isinstance(frame, bytes):
                frame = memoryview(frame)

            assert frame.contiguous, "Only contiguous arrays are supported"
            struct.pack_into("<I", shm_buffer.buf, offset + total_size, frame.nbytes)
            total_size += struct.calcsize("<I")
            shm_buffer.buf[offset + total_size : offset + total_size + frame.nbytes] = frame.cast("B")

            total_size += frame.nbytes

        struct.pack_into("<I", shm_buffer.buf, offset, total_size)
        return total_size

    def _handle_frames(self, memory_name, block_offset: int) -> List[memoryview]:

        if self._shm_buffer and self._shm_buffer.name != memory_name:
            self.dispose()
        if self._shm_buffer is None:
            self._shm_buffer = SharedMemory(memory_name, create=False)

        if self._shm_buffer.size < block_offset:
            raise RuntimeError(f"Shared memory buffer is too small for requested offset {block_offset}")

        frames = []
        (total_size,) = struct.unpack_from("<I", self._shm_buffer.buf, block_offset)
        offset = struct.calcsize("<I")
        while offset < total_size:
            (frame_size,) = struct.unpack_from("<I", self._shm_buffer.buf, block_offset + offset)
            offset += struct.calcsize("<I")
            frame = self._shm_buffer.buf[block_offset + offset : block_offset + offset + frame_size]
            offset += frame_size
            frames.append(frame)
        return frames

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
        self._memory_name = None
