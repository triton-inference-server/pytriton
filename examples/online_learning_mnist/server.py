#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Server shows online learning model concept."""
import argparse
import logging
import threading
from queue import Queue
from threading import Lock

import numpy as np
import torch  # pytype: disable=import-error
import torch.nn.functional as functional  # pytype: disable=import-error
import torch.optim as optim  # pytype: disable=import-error
from torch.optim.lr_scheduler import StepLR  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from model import Net  # pytype: disable=import-error # isort:skip

LOGGER = logging.getLogger("examples.online_learning_mnist.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


class Trainer:
    """Trainer class for MNIST model.
    It is used to train the model and to keep track of the training progress.
    It defines the learning rate scheduler and optimizer.
    It organizes the training process in epochs.
    """

    def __init__(self, model, lr, gamma, epoch_size):
        self.model = model
        self.optimizer = optim.Adadelta(model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=gamma)
        self.iter = 0
        self.epoch = 0
        self.epoch_size = epoch_size

    def train_batch(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = functional.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        self.iter += 1
        return loss.item()

    def ready(self):
        return self.iter >= self.epoch_size

    def next_epoch(self):
        self.iter = 0
        self.epoch += 1
        self.scheduler.step()


class OnlineLearning(threading.Thread):
    """Online learning class that implements two infer functions: train and infer.
    Infer function is used in inference endpoint and train function is used in training endpoint.
    Train function collects data and trains model in background thread.
    Infer function uses trained model to make inference.
    When trained model is ready, it is swapped with infer model.
    """

    def __init__(self, device, lr, gamma, epoch_size, max_queue_size):
        super().__init__()
        self.device = device

        self.trained_model = Net().to(self.device)
        self.trained_model.train()
        self.infer_model = Net().to(self.device)
        self.infer_model.eval()
        self.stopped = False
        self.train_data_queue = Queue(maxsize=max_queue_size)

        self.lock = Lock()
        self.trainer = Trainer(self.trained_model, lr, gamma, epoch_size)
        self.last_loss = 0.0

    def run(self) -> None:
        while not self.stopped:
            image, target = self.train_data_queue.get()
            if self.stopped:
                return

            data_tensor = torch.from_numpy(image).to(self.device)
            labels = target.reshape((target.shape[0],))
            labels_tensor = torch.from_numpy(labels).to(self.device)
            self.last_loss = self.trainer.train_batch(data_tensor, labels_tensor)

            if self.trainer.ready():
                self.replace_inference_model()
                self.trainer.next_epoch()

    def stop(self):
        self.stopped = True
        self.train_data_queue.put((None, None))
        self.join()

    def replace_inference_model(self):
        with self.lock:
            self.infer_model.load_state_dict(self.trained_model.state_dict())

    def train(self, requests):
        """Train function is used in training endpoint."""
        # concatenate all requests into one batch. No need for padding due to fixed image dimensions
        images = np.concatenate([request["image"] for request in requests], axis=0)
        targets = np.concatenate([request["target"] for request in requests], axis=0)
        self.train_data_queue.put((images, targets))
        return [{"last_loss": np.array([[self.last_loss]]).astype(np.float32)} for _ in requests]

    @batch
    def infer(self, image):
        """Infer function is used in inference endpoint."""
        data_tensor = torch.from_numpy(image).to(self.device)
        with self.lock:
            res = self.infer_model(data_tensor)
        res = res.numpy(force=True)
        return {"predictions": res}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging in debug mode.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    log_verbose = 1 if args.verbose else 0
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    online_learning_model = OnlineLearning(
        device=torch.device("cuda"), lr=1.0, gamma=0.7, epoch_size=134, max_queue_size=1000
    )
    online_learning_model.start()
    try:
        with Triton(config=TritonConfig(log_verbose=log_verbose)) as triton:
            LOGGER.info("Loading OnlineLearning model")
            triton.bind(
                model_name="MnistTrain",
                infer_func=online_learning_model.train,
                inputs=[
                    # image for training
                    Tensor(name="image", dtype=np.float32, shape=(1, 28, 28)),
                    # target class corresponding to image (class index from 0 to 9)
                    Tensor(name="target", dtype=np.int64, shape=(1,)),
                ],
                outputs=[
                    # last loss value batch
                    Tensor(name="last_loss", dtype=np.float32, shape=(1,)),
                ],
                config=ModelConfig(max_batch_size=64),
                strict=True,
            )
            triton.bind(
                model_name="MnistInfer",
                infer_func=online_learning_model.infer,
                inputs=[
                    # image for classification
                    Tensor(name="image", dtype=np.float32, shape=(1, 28, 28)),
                ],
                outputs=[
                    # predictions taken from softmax layer
                    Tensor(name="predictions", dtype=np.float32, shape=(-1,)),
                ],
                config=ModelConfig(max_batch_size=64),
                strict=True,
            )

            LOGGER.info("Serving model")
            triton.serve()
    finally:
        LOGGER.info("Stopping online learning model")
        online_learning_model.stop()


if __name__ == "__main__":
    main()
