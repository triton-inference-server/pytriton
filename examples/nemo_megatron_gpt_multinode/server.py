#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
"""Text generation server with NeMo Megatron GPT model."""

import torch  # pytype: disable=import-error
from nemo.collections.nlp.modules.common.text_generation_utils import generate  # pytype: disable=import-error
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin  # pytype: disable=import-error
from pytorch_lightning.trainer.trainer import Trainer  # pytype: disable=import-error

from pytriton.model_config import ModelConfig
from pytriton.triton import Triton, TritonConfig

from gpt import NemoGptCallable  # pytype: disable=import-error # isort:skip
from helpers import download_and_load_model, setup_distributed_environment  # pytype: disable=import-error # isort:skip

if not torch.cuda.is_available():
    raise OSError("GPU is needed for the inference")

ENDPOINT_BIND_ADDRESS = "0.0.0.0"
HTTP_PORT = 8000


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        default="-1",
        help=(
            "Number of GPUs to load model on or exact identifiers of GPUs to use separated by comma. "
            "If set to -1 all available GPU will be used."
        ),
    )
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to load model on")
    parser.add_argument(
        "--model-repo-id", default="nvidia/nemo-megatron-gpt-1.3B", help="Model repository id on HuggingFace Hub"
    )
    args = parser.parse_args()

    trainer = Trainer(
        plugins=NLPDDPPlugin(),
        devices=args.gpus,
        num_nodes=args.nodes,
        accelerator="gpu",
        logger=False,
        precision=16,
    )

    model = download_and_load_model(args.model_repo_id, trainer)
    app_state = setup_distributed_environment(trainer)
    if app_state.global_rank == 0:
        infer_callable = NemoGptCallable(model_name="GPT", model=model)
        triton_config = TritonConfig(http_address=ENDPOINT_BIND_ADDRESS, http_port=HTTP_PORT, log_verbose=4)
        with Triton(config=triton_config) as triton:
            triton.bind(
                model_name=infer_callable.model_name,
                infer_func=infer_callable.infer,
                inputs=infer_callable.inputs,
                outputs=infer_callable.outputs,
                config=ModelConfig(max_batch_size=128),
            )
            triton.serve()
    else:
        while True:
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 0:
                generate(model.cuda())


if __name__ == "__main__":
    main()
