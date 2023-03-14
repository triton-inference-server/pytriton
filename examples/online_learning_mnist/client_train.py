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
"""Client for online_learning sample server."""
import logging

import torch  # pytype: disable=import-error
from torchvision import datasets, transforms  # pytype: disable=import-error

from pytriton.client import ModelClient

LOGGER = logging.getLogger("examples.online_learning_mnist.client_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    epoch = 0
    epoch_size = 134
    with ModelClient("localhost", "MnistTrain") as client:
        LOGGER.info("Training:")
        for _ in range(2):
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx % epoch_size == 0:
                    LOGGER.info(f"Epoch: {epoch}")
                    epoch += 1
                data = data.numpy()
                target = target.numpy()
                target = target.reshape((target.shape[0], 1))

                # In this example, train inference returns the laste training loss in 'results' array
                client.infer_batch(image=data, target=target)


if __name__ == "__main__":
    main()
