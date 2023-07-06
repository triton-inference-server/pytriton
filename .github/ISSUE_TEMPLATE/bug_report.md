---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Description**

A clear and concise description of the bug.

**To reproduce**

If relevant, add a minimal example so that we can reproduce the error, if necessary, by running the code. For example:

```python
# server
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

@batch
def _infer_fn(**inputs):
    ...
    results_dict = model(**inputs)  # ex note: the bug is here, we expect to receive ...
    ...
    # note: observing results_dict as dictionary of numpy arrays
    return results_dict


with Triton() as triton:
    triton.bind(
        model_name="MyModel",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="in1", dtype=np.float32, shape=(-1,)),
            Tensor(name="in2", dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="out1", dtype=np.float32, shape=(-1,)),
            Tensor(name="out2", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    triton.serve()
```

```python
# client
import numpy as np
from pytriton.client import ModelClient

batch_size = 2
in1_batch = np.ones((batch_size, 1), dtype=np.float32)
in2_batch = np.ones((batch_size, 1), dtype=np.float32)

with ModelClient("localhost", "MyModel") as client:
    result_batch = client.infer_batch(in1_batch, in2_batch)
```

**Observed results and expected behavior**

Please describe the observed results as well as the expected results.
If possible, attach relevant log output to help analyze your problem.
If an error is raised, please paste the full traceback of the exception.

```

```

**Environment**

- OS/container version: [e.g., container nvcr.io/nvidia/pytorch:23.02-py3 / virtual machine with Ubuntu 22.04]
  - glibc version: [e.g., 2.31; can be checked with `ldd --version`]
- Python interpreter distribution and version: [e.g., CPython 3.8 / conda 4.7.12 with Python 3.8 environment]
- pip version: [e.g., 23.1.2]
- PyTriton version: [e.g., 0.1.4 / custom build from source at commit ______]
- Deployment details: [e.g., multi-node multi-GPU setup on GKE / multi-GPU single-node setup in Jupyter Notebook]

**Additional context**
Add any other context about the problem here.
