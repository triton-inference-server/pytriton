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
"""Module with logging related utils."""

import logging


def silence_3rd_party_loggers():
    """Silence 3rd party libraries which adds enormous number of log lines on DEBUG level."""
    logging.getLogger("sh.command").setLevel(logging.WARNING)
    logging.getLogger("sh.stream_bufferer").setLevel(logging.WARNING)
    logging.getLogger("sh.streamreader").setLevel(logging.WARNING)
