# noqa: CPY001
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ---------------------------------------------------------------------------------------------------- #
# This file is an excerpt from CV-CUDA segmentation example:                                           #
# https://github.com/CVCUDA/CV-CUDA/blob/release_v0.3.x/samples/segmentation/python/model_inference.py #
# ---------------------------------------------------------------------------------------------------- #

import logging

import nvtx  # pytype: disable=import-error
import torch  # pytype: disable=import-error
from torchvision.models import segmentation as segmentation_models  # pytype: disable=import-error


class SegmentationPyTorch:
    def __init__(self, seg_class_name, device_id):
        self.logger = logging.getLogger(__name__)
        self.device_id = device_id
        # Fetch the segmentation index to class name information from the weights
        # meta properties.
        # The underlying pytorch model that we use for inference is the FCN model
        # from torchvision.
        torch_model = segmentation_models.fcn_resnet101
        weights = segmentation_models.FCN_ResNet101_Weights.DEFAULT

        try:
            self.class_index = weights.meta["categories"].index(seg_class_name)
        except ValueError:
            raise ValueError(
                f"Requested segmentation class '{seg_class_name}' is not supported by the "
                f"fcn_resnet101 model. All supported class names are: {', '.join(weights.meta['categories'])}"
            ) from None

        # Inference uses PyTorch to run a segmentation model on the pre-processed
        # input and outputs the segmentation masks.
        class FCN_Softmax(torch.nn.Module):  # noqa: N801
            def __init__(self, fcn):
                super().__init__()
                self.fcn = fcn

            def forward(self, x):
                infer_output = self.fcn(x)["out"]
                return torch.nn.functional.softmax(infer_output, dim=1)

        fcn_base = torch_model(weights=weights)
        fcn_base.eval()
        self.model = FCN_Softmax(fcn_base).cuda(self.device_id)
        self.model.eval()

        self.logger.info("Using PyTorch as the inference engine.")

    def __call__(self, tensor):
        nvtx.push_range("inference.torch")

        with torch.no_grad():
            segmented = self.model(tensor)

        nvtx.pop_range()
        return segmented
