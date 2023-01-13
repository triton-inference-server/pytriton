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
Runs inference session over NLP model
"""

import logging
import pathlib
import tempfile
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from tests.functional.common.models import Framework, TestModelSpec

logger = logging.getLogger(__package__)


def huggingface_distilbert(test_time_s: int, init_timeout_s: int, batch_size: int, sequence_length: int, verbose: bool):
    import tensorflow  # pytype: disable=import-error

    from pytriton.client import ModelClient
    from pytriton.triton import Triton, TritonConfig
    from tests.utils import find_free_port

    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    model_name = "distilbert-base-uncased"

    model_spec = _model_spec()
    dataset = _dataset(
        model_name=model_name,
        dataset_name="imdb",
        sequence_length=sequence_length,
        input_names=[inpt.name for inpt in model_spec.inputs],
        batch_size=1,
    )

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

                def infer_model():
                    thread_id = threading.get_ident()
                    url = f"http://localhost:{triton_config.http_port}"
                    with ModelClient(url, model_spec.name, init_timeout_s=init_timeout_s) as client:

                        import time

                        should_stop_at_s = time.time() + test_time_s
                        for idx, data_tensor in enumerate(dataset):
                            input_ids = data_tensor["input_ids"].numpy()
                            attention_mask = data_tensor["attention_mask"].numpy()
                            logger.debug(f"Request send from {thread_id}")

                            result = client.infer_batch(input_ids=input_ids, attention_mask=attention_mask)
                            if idx > 0 and idx % 10 == 0:
                                time_left_s = max(should_stop_at_s - time.time(), 0.0)
                                logger.debug(
                                    f"[{thread_id}] Processed {idx} batches time left: {time_left_s:0.1f}s \n."
                                    f"Result: {len(result)}."
                                )

                                if time_left_s <= 0:
                                    break

                infer_threads = [infer_model] * batch_size
                with ThreadPoolExecutor() as executor:
                    running_tasks = [executor.submit(infer_task) for infer_task in infer_threads]
                    for running_task in running_tasks:
                        running_task.result()

        finally:
            if triton_log_path.exists():
                logger.debug("-" * 64)
                server_logs = triton_log_path.read_text(errors="replace")
                server_logs = "--- triton logs:\n\n" + textwrap.indent(server_logs, prefix=" " * 8)
                logger.debug(server_logs)


def _create_hf_tensorflow_distilbert_base_uncased_fn(model_name: str) -> Callable:
    from transformers.models.distilbert.modeling_tf_distilbert import (  # pytype: disable=import-error
        TFDistilBertForMaskedLM,
    )

    model = TFDistilBertForMaskedLM.from_pretrained(model_name)
    model.config.return_dict = True
    model.config.use_cache = False

    @batch
    def _infer_fn(input_ids, attention_mask):
        logger.debug(f"input_ids: {input_ids.shape}")
        logger.debug(f"attention_mask: {attention_mask.shape}")
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
