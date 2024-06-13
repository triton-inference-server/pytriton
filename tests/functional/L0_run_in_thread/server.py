#!/usr/bin/env python3
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
"""e2e run triton in not main thread"""

import argparse
import logging
import random
import signal
import threading
import traceback
from typing import Any

from pytriton.check.utils import find_free_port
from pytriton.triton import Triton, TritonConfig
from tests.functional.common.models import ADD_SUB_PYTHON_MODEL

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])

signal_handled = False
cond = threading.Condition()


class MyTritonThread(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        self.triton_config = None
        self.exception_traceback = None
        self.triton = None
        self.args = args

    def run(self) -> None:
        try:
            assert self.args is not None
            assert self.args.grpc_port is not None
            assert self.args.http_port is not None

            self.triton_config = TritonConfig(
                grpc_port=self.args.grpc_port, http_port=self.args.http_port, metrics_port=find_free_port()
            )
            LOGGER.debug(f"Using {self.triton_config}")
            self.triton = Triton(config=self.triton_config)
            model_spec = ADD_SUB_PYTHON_MODEL
            LOGGER.debug(f"Using {model_spec}")
            self.triton.bind(
                model_name=model_spec.name,
                infer_func=model_spec.create_infer_fn(),
                inputs=model_spec.inputs,
                outputs=model_spec.outputs,
                config=model_spec.model_config,
            )
            self.triton.serve()

        except Exception:
            self.exception_traceback = traceback.format_exc()
            with cond:
                cond.notify()


def signal_handler(_signal_num: Any, _) -> None:
    with cond:
        global signal_handled
        signal_handled = True
        cond.notify()


def main():
    from pytriton.check.utils import DEFAULT_LOG_FORMAT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Size of single inference batch")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Timeout for test")
    parser.add_argument("--grpc-port", type=int, help="Grpc triton port")
    parser.add_argument("--http-port", type=int, help="Http triton port")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)
    LOGGER.debug(f"CLI args: {args}")

    random.seed(args.seed)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    th = MyTritonThread(args)
    th.start()

    with cond:
        cond.wait()

    assert signal_handled
    assert th.triton is not None
    th.triton.stop()
    LOGGER.info("Signal handled and triton server properly stopped")

    assert th.exception_traceback is None, f"Raised {th.exception_traceback}"


if __name__ == "__main__":
    main()
