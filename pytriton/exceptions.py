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
"""pytriton exceptions definition."""


class PytritonError(Exception):
    """Generic pytriton exception."""

    def __init__(self, message: str):
        """Initialize exception with message.

        Args:
            message: Error message
        """
        self._message = message

    def __str__(self) -> str:
        """Return exception as a string.

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


class PytritonValidationError(PytritonError):
    """pytriton configuration validation exception."""

    pass


class PytritonInvalidOperationError(PytritonError):
    """pytriton invalid operation exception."""

    pass


class PytritonBadParameterError(PytritonError):
    """pytriton invalid parameter exception."""

    pass


class PytritonModelConfigError(PytritonError):
    """pytriton invalid model config exception."""

    pass


class PytritonUnrecoverableError(PytritonError):
    """Unrecoverable error occurred in inference function, thus no further inferences possible."""

    pass


class PytritonRuntimeError(PytritonError):
    """Raised when an error is detected that doesn’t fall in any of the other categories."""

    pass
