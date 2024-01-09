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
"""Common data structures and type used by proxy model and inference handler."""

import dataclasses
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclasses.dataclass
class Request:
    """Data class for request data including numpy array inputs."""

    data: Dict[str, np.ndarray]
    """Input data for the request."""
    parameters: Optional[Dict[str, Union[str, int, bool]]] = None
    """Parameters for the request."""

    def __getitem__(self, input_name: str) -> np.ndarray:
        """Get input data."""
        return self.data[input_name]

    def __setitem__(self, input_name: str, input_data: np.ndarray):
        """Set input data."""
        self.data[input_name] = input_data

    def __delitem__(self, input_name: str):
        """Delete input data from request."""
        del self.data[input_name]

    def __len__(self):
        """Get number of inputs."""
        return len(self.data)

    def __iter__(self):
        """Iterate over input names."""
        return iter(self.data)

    def items(self):
        """Iterate over input names and data."""
        return self.data.items()

    def keys(self):
        """Iterate over input names."""
        return self.data.keys()

    def values(self):
        """Iterate over input data."""
        return self.data.values()


Requests = List[Request]


@dataclasses.dataclass
class Response:
    """Data class for response data including numpy array outputs."""

    data: Dict[str, np.ndarray]

    def __getitem__(self, output_name: str) -> np.ndarray:
        """Get output data."""
        return self.data[output_name]

    def __setitem__(self, output_name: str, output_data: np.ndarray):
        """Set output data."""
        self.data[output_name] = output_data

    def __delitem__(self, output_name: str):
        """Delete output data from response."""
        del self.data[output_name]

    def __len__(self):
        """Get number of outputs."""
        return len(self.data)

    def __iter__(self):
        """Iterate over output names."""
        return iter(self.data)

    def items(self):
        """Iterate over output names and data."""
        return self.data.items()

    def keys(self):
        """Iterate over output names."""
        return self.data.keys()

    def values(self):
        """Iterate over output data."""
        return self.data.values()


Responses = List[Response]
ResponsesOrError = Union[Responses, Exception]
ResponsesNoneOrError = Union[Responses, None, Exception]

Scope = Dict[str, Any]
