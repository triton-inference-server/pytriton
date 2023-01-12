#!/usr/bin/env python3
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
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import torch  # pytype: disable=import-error

from pytriton.client import ModelClient

logger = logging.getLogger("examples.multi_instance_linear_pytorch.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def infer_model():
    thread_id = threading.get_ident()
    with ModelClient("localhost", "Linear") as client:
        input_batch = torch.randn(128, 20).cpu().detach().numpy()
        logger.info(f"Batch: {input_batch.tolist()}")

        for idx in range(25):
            request_idx = idx + 1
            logger.info(f"Sending inference request {request_idx} and thread {thread_id}.")
            result_batch = client.infer_batch(input_batch)

            logger.info(f"Result of size: {len(result_batch)} for request {request_idx} and thread {thread_id}.")


with ThreadPoolExecutor() as executor:
    running_tasks = [executor.submit(infer_task) for infer_task in [infer_model, infer_model]]
    for running_task in running_tasks:
        running_task.result()
