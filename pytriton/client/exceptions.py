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
"""Exceptions thrown in pytriton.client module."""


class PytritonClientError(Exception):
    """Generic pytriton client exception."""

    def __init__(self, message: str):
        """Initialize exception with message.

        Args:
            message: Error message
        """
        self._message = message

    def __str__(self) -> str:
        """String representation of error.

        Returns:
            Message content
        """
        return self._message

    @property
    def message(self):
        """Get the exception message.

        Returns:
            The message associated with this exception, or None if no message.

        """
        return self._message


class PytritonClientValueError(PytritonClientError):
    """Generic error raised in case of incorrect values are provided into API."""

    pass


class PytritonClientUrlParseError(PytritonClientValueError):
    """Error raised on problems with parsing Triton Inference Server url."""

    pass


class PytritonClientTimeoutError(PytritonClientError):
    """Timeout occurred during communication with the Triton Inference Server."""

    pass


class PytritonClientModelUnavailableError(PytritonClientError):
    """Model with given name and version is unavailable on the given Triton Inference Server."""

    pass


class PytritonClientModelDoesntSupportBatchingError(PytritonClientError):
    """Error raised in case of trying to infer batch on model not supporting batching."""

    pass


class PytritonClientInferenceServerError(PytritonClientError):
    """Error raised in case of error on inference function or Triton Inference Server side."""

    pass
