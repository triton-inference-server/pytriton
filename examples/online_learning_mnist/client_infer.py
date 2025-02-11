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

import argparse
import logging

import torch  # pytype: disable=import-error
import torch.nn.functional as functional  # pytype: disable=import-error
from torchvision import datasets, transforms  # pytype: disable=import-error

from pytriton.client import ModelClient

LOGGER = logging.getLogger("examples.online_learning_mnist.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def main():
    global args
    parser = argparse.ArgumentParser(description="Inference client")
    parser.add_argument("--iter", required=False, default=300, type=int, help="Number of iterations to run")
    args = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset2 = datasets.MNIST("../data", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64)
    LOGGER.info("Inference results:")
    with ModelClient("localhost", "MnistInfer") as client:
        with torch.no_grad():
            for _ in range(args.iter):
                test_loss = 0
                correct = 0
                for _batch_idx, (data, target) in enumerate(test_loader):
                    data_np = data.numpy()
                    inference_results = client.infer_batch(image=data_np)
                    prediction_np = inference_results["predictions"]
                    prediction = torch.from_numpy(prediction_np)

                    test_loss += functional.nll_loss(prediction, target, reduction="sum").item()  # sum up batch loss
                    pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(test_loader.dataset)
                LOGGER.info(
                    "\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%)\n",
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )


if __name__ == "__main__":
    main()
