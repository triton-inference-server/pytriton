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
import dataclasses
import enum
import subprocess
from typing import Callable, Optional


class Framework(enum.Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    framework: Optional[Framework]
    create_dataset_fn: Callable


COCO_LABELS = {
    # 0: 'background',
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def _create_tfds_coco2017_validation(batch_size: Optional[int] = None) -> Callable:
    subprocess.run(["pip", "install", "--upgrade", "tensorflow-datasets"], check=True)

    import tensorflow_datasets as tfds  # pytype: disable=import-error

    return tfds.load("coco/2017", split="validation", as_supervised=True, with_info=True, batch_size=batch_size)


TFDS_COCO2017_VALIDATION_DATASET = DatasetSpec(
    framework=Framework.TENSORFLOW,
    create_dataset_fn=_create_tfds_coco2017_validation,
)


def _create_tfds_tf_flowers(batch_size: Optional[int] = None):
    subprocess.run(["pip", "install", "--upgrade", "tensorflow-datasets"], check=True)

    import tensorflow_datasets as tfds  # pytype: disable=import-error

    return tfds.load("tf_flowers", as_supervised=True, with_info=True, batch_size=batch_size)


TFDS_TF_FLOWERS_DATASET = DatasetSpec(
    framework=Framework.TENSORFLOW,
    create_dataset_fn=_create_tfds_tf_flowers,
)


DATASETS_CATALOGUE = [TFDS_COCO2017_VALIDATION_DATASET, TFDS_TF_FLOWERS_DATASET]
