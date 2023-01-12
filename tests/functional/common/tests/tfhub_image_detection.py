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
"""
Runs inference session over image detector model
"""
import collections
import logging
import pathlib
import tempfile
import textwrap
import time


def tfhub_image_detection(test_time_s: int, init_timeout_s: int, batch_size: int, verbose: bool):
    from pytriton.client import ModelClient
    from pytriton.triton import Triton, TritonConfig
    from tests.functional.common.datasets import COCO_LABELS, TFDS_TF_FLOWERS_DATASET
    from tests.functional.common.models import EFFICIENTDET_DETECTION_TF_MODEL
    from tests.utils import find_free_port

    logger = logging.getLogger(__package__)

    model_spec = EFFICIENTDET_DETECTION_TF_MODEL
    dataset_spec = TFDS_TF_FLOWERS_DATASET
    expected_5_most_common_coco_labels = ("potted plant", "vase", "person", "bird", "tv")

    infer_fn = model_spec.create_infer_fn()
    (dataset, dataset_info) = dataset_spec.create_dataset_fn(batch_size=batch_size)
    classes = COCO_LABELS

    with tempfile.TemporaryDirectory() as temp_dir:
        triton_log_path = pathlib.Path(temp_dir) / "triton.log"
        try:
            triton_config = TritonConfig(
                grpc_port=find_free_port(),
                http_port=find_free_port(),
                metrics_port=find_free_port(),
                log_verbose=int(verbose),
                log_file=triton_log_path,
            )
            with Triton(config=triton_config) as triton:
                triton.bind(
                    model_name=model_spec.name,
                    infer_func=infer_fn,
                    inputs=model_spec.inputs,
                    outputs=model_spec.outputs,
                    config=model_spec.model_config,
                )
                triton.run()

                classes_counter = collections.Counter()
                url = f"http://localhost:{triton_config.http_port}"
                with ModelClient(url, model_spec.name, init_timeout_s=init_timeout_s) as client:

                    should_stop_at_s = time.time() + test_time_s
                    dataset = dataset["train"].repeat()
                    for idx, (image_tensor, _) in enumerate(dataset):
                        image_data = image_tensor.numpy()  # padded with 0
                        result = client.infer_batch(image=image_data)
                        batch_classes = [
                            result["classes"][idx, result["scores"][idx] >= 0.3]
                            for idx in range(result["scores"].shape[0])
                        ]
                        classes_counter.update([classes[clazz] for classes_ in batch_classes for clazz in classes_])
                        if idx > 0 and idx % 10 == 0:
                            time_left_s = max(should_stop_at_s - time.time(), 0.0)
                            logger.debug(
                                f"Processed {idx} batches time left: {time_left_s:0.1f}s "
                                f"most common labels: {classes_counter.most_common(5)} "
                                f"expected common labels: {expected_5_most_common_coco_labels}"
                            )

                            most_common = tuple(clazz for clazz, n in classes_counter.most_common(5))
                            assert (
                                len(set(most_common) ^ set(expected_5_most_common_coco_labels)) <= 4
                            ), f"difference on {set(most_common) ^ set(expected_5_most_common_coco_labels)}"
                            if time_left_s <= 0:
                                break

        finally:
            if triton_log_path.exists():
                logger.debug("-" * 64)
                server_logs = triton_log_path.read_text(errors="replace")
                server_logs = "--- triton logs:\n\n" + textwrap.indent(server_logs, prefix=" " * 8)
                logger.debug(server_logs)
