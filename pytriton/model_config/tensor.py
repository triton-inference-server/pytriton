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
"""Tensor object definition.

Describe the model input or output.

    Examples of use:

        # Minimal constructors
        tensor = Tensor(dtype=np.bytes_, shape=(-1,))
        tensor = Tensor(dtype=np.float32, shape=(-1,))

        # Type definition from existing object
        a = np.array([1, 2, 3, 4])
        tensor = Tensor(dtype=a.dtype, shape=(-1,))

        # Custom name
        tensor = Tensor(name="data", dtype=np.float32, shape=(16,))
"""

import dataclasses
from typing import Optional, Type, Union

import numpy as np


@dataclasses.dataclass(frozen=True)
class Tensor:
    """Model input and output definition for Triton deployment.

    Args:
        shape: Shape of the input/output tensor.
        dtype: Data type of the input/output tensor.
        name: Name of the input/output of model.
        optional: Flag to mark if input is optional.
    """

    shape: tuple
    dtype: Union[np.dtype, Type[np.dtype], Type[object]]
    name: Optional[str] = None
    optional: Optional[bool] = False

    def __post_init__(self):
        """Override object values on post init or field override."""
        if isinstance(self.dtype, np.dtype):
            object.__setattr__(self, "dtype", self.dtype.type)  # pytype: disable=attribute-error
