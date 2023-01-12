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

# pytype: disable=import-error
import pathlib
import socket

import filelock
import huggingface_hub
import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils.app_state import AppState
from pytorch_lightning.trainer.trainer import Trainer
from server_impl import MegatronTritonServer

# pytype: enable=import-error


if not torch.cuda.is_available():
    raise OSError("GPU is needed for the inference")


HTTP_PORT = 8000


def _setup_distributed_environment(trainer):
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    app_state = AppState()

    hostname = socket.gethostname()
    print(  # noqa
        f"global={app_state.global_rank}/{app_state.world_size} "
        f"local={app_state.local_rank} @ {hostname}:{app_state.device_id} / "
        f"dp={app_state.data_parallel_rank}/{app_state.data_parallel_size} "
        f"tp={app_state.tensor_model_parallel_rank}/{app_state.tensor_model_parallel_size} "
        f"pp={app_state.pipeline_model_parallel_rank}/{app_state.pipeline_model_parallel_size} "
        "vpp="
        f"{getattr(app_state, 'virtual_pipeline_model_parallel_rank', None)}/"
        f"{getattr(app_state, 'virtual_pipeline_model_parallel_size', None)}"
    )
    return app_state


def _bind(repo_id, trainer):
    lock = filelock.FileLock("/tmp/nemo_megatron_gpt.lock")
    with lock:
        repo_dir_path = huggingface_hub.snapshot_download(repo_id)
        model_path = list(pathlib.Path(repo_dir_path).rglob("*.nemo"))[0]
        # although putting model load in filelock section might significantly increase load time
        # especially while loading large models in slurm multi-node scenario
        # but there might be tokenizer files download which is not distributed jobs safe
        model = MegatronGPTModel.restore_from(restore_path=model_path.as_posix(), trainer=trainer)

    model.freeze()
    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    return model


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
        plugins=NLPDDPPlugin(), devices=args.gpus, num_nodes=args.nodes, accelerator="gpu", logger=False, precision=16
    )

    model = _bind(args.model_repo_id, trainer)
    app_state = _setup_distributed_environment(trainer)
    if app_state.global_rank == 0:
        print(f"Server http url: http://{socket.gethostname()}:{HTTP_PORT}")  # noqa
        megatron_server = MegatronTritonServer(model.cuda())
        megatron_server.serve("0.0.0.0", http_port=HTTP_PORT)
    else:
        while True:
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 0:
                generate(model.cuda())


if __name__ == "__main__":
    main()
