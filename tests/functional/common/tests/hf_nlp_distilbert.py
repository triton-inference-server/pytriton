# Copyright (c) 2022-23, NVIDIA CORPORATION. All rights reserved.
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
Runs inference session over NLP model
"""

import logging
import pathlib
import tempfile
from concurrent.futures import FIRST_COMPLETED, wait
from typing import Callable, List

import numpy as np

from pytriton.check.utils import find_free_port
from pytriton.client import FuturesModelClient
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from tests.functional.common.models import Framework, TestModelSpec

logger = logging.getLogger(__package__)


def huggingface_distilbert(test_time_s: int, init_timeout_s: int, batch_size: int, sequence_length: int, verbose: bool):
    import tensorflow  # pytype: disable=import-error

    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    model_name = "distilbert-base-uncased"

    model_spec = _model_spec()

    logger.debug("generating dataset")
    dataset = _dataset(
        model_name=model_name,
        dataset_name="imdb",
        sequence_length=sequence_length,
        input_names=[inpt.name for inpt in model_spec.inputs],
        batch_size=1,
    )

    def requests_generator():
        for data_tensor in dataset:
            input_ids = data_tensor["input_ids"].numpy()
            attention_mask = data_tensor["attention_mask"].numpy()
            for _ in range(batch_size):
                yield {"input_ids": input_ids, "attention_mask": attention_mask}

    requests = list(requests_generator())

    logger.debug("data generated")

    infer_fn = model_spec.create_infer_fn(model_name=model_name)
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

                logger.debug("Triton server started")

                # Send requests
                url = f"http://localhost:{triton_config.http_port}"
                with FuturesModelClient(url, model_spec.name, max_workers=batch_size) as client:
                    # Wait for model
                    client.wait_for_model(init_timeout_s).result()

                    import time

                    should_stop_at_s = time.time() + test_time_s

                    number_of_processed_requests = 0

                    not_done = {*()}
                    for request in requests:
                        result_future = client.infer_batch(**request)
                        not_done.add(result_future)
                        if len(not_done) > batch_size:
                            done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
                            if len(done) > 0:
                                future = done.pop()
                                result = future.result()
                                number_of_processed_requests += len(done)
                                if number_of_processed_requests > 0 and number_of_processed_requests % 10 == 0:
                                    time_left_s = max(should_stop_at_s - time.time(), 0.0)
                                    logger.debug(
                                        "Processed %d batches time left: %.1fs \n Result: %d.",
                                        number_of_processed_requests,
                                        time_left_s,
                                        len(result),
                                    )
                        time_left_s = max(should_stop_at_s - time.time(), 0.0)
                        if time_left_s <= 0:
                            break
                logger.debug("Test finished")

        finally:
            if triton_log_path.exists():
                logger.info("%s triton logs %s", "-" * 32, "-" * 32)
                triton_logs_logger = logger.getChild("triton_logs")
                stdout_handler = logging.StreamHandler()
                stdout_handler.setFormatter(logging.Formatter("        %(message)s"))
                triton_logs_logger.addHandler(stdout_handler)
                triton_logs_logger.propagate = False
                for line in triton_log_path.read_text(errors="replace").splitlines():
                    triton_logs_logger.info(line)


def _create_hf_tensorflow_distilbert_base_uncased_fn(model_name: str) -> Callable:
    import tensorflow as tf
    from transformers.models.distilbert.modeling_tf_distilbert import (  # pytype: disable=import-error
        TFDistilBertForMaskedLM,
    )

    model = TFDistilBertForMaskedLM.from_pretrained(model_name, use_safetensors=False)
    model.config.return_dict = True
    model.config.use_cache = False

    @batch
    def _infer_fn(input_ids, attention_mask):
        logger.debug("input_ids: %s", input_ids.shape)
        logger.debug("attention_mask: %s", attention_mask.shape)
        device = "/GPU:0"  # change this to the GPU device you want to use
        with tf.device(device):
            result = model(input_ids, attention_mask)
        return {"logits": result.logits.numpy()}

    return _infer_fn


def _model_spec() -> TestModelSpec:
    model_spec = TestModelSpec(
        name="DistilBert",
        framework=Framework.TENSORFLOW,
        create_infer_fn=_create_hf_tensorflow_distilbert_base_uncased_fn,
        inputs=(
            Tensor(name="input_ids", dtype=np.int64, shape=(-1,)),
            Tensor(name="attention_mask", dtype=np.int64, shape=(-1,)),
        ),
        outputs=(
            Tensor(
                name="logits",
                dtype=np.float32,
                shape=(-1, -1),
            ),
        ),
        model_config=ModelConfig(
            max_batch_size=16,
            batcher=DynamicBatcher(
                max_queue_delay_microseconds=5000,
            ),
        ),
    )
    return model_spec


def _dataset(model_name: str, dataset_name: str, sequence_length: int, input_names: List[str], batch_size: int):
    from datasets import load_dataset  # pytype: disable=import-error
    from transformers import AutoTokenizer, DataCollatorWithPadding, TensorType  # pytype: disable=import-error

    dataset = load_dataset(dataset_name)["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _preprocess_text_dataset(examples):
        return tokenizer(examples["text"], truncation=True, max_length=sequence_length)

    tokenized_dataset = dataset.map(_preprocess_text_dataset, batched=True)
    dataset = tokenized_dataset.remove_columns([c for c in tokenized_dataset.column_names if c not in input_names])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=sequence_length,
        return_tensors=TensorType.NUMPY,
    )

    return dataset.to_tf_dataset(
        columns=dataset.column_names,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
