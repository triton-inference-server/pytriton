<!--
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Using PyTriton Check Tool

The PyTriton Check Tool is a command-line utility included with PyTriton, designed to perform preliminary checks on the environment where PyTriton is deployed. It collects vital information about the operating system version, socket statistics, disk and memory usage and GPU functionality, ensuring the environment is properly set up for PyTriton.

## Features

- **Operating System Verification:** Collects and verifies the OS version to ensure compatibility.
- **Python version and compiler:** Gathers information about python version, libc used in python and installed compiler version.
- **Socket Statistics:** Gathers socket statistics for network-related checks.
- **Disk/Memory/CPU Statistics:** Gathers information about disk, memory, vpu usage
- **GPU Functionality:** Runs `nvidia-smi` to verify GPU functionality.
- **Inference Testing:** Executes simple inference examples on PyTriton to ensure proper communication with the Triton Inference Server.

## How to Use

To launch the tool, use one of the following commands:

<!--pytest.mark.skip-->
```bash
pytriton check -w /path/to/non_existing_workspace_dir

# or

python -m pytriton check -w /path/to/non_existing_workspace_dir
```

### Command Explanation

- `-w /path/to/non_existing_workspace_dir`: Specifies the workspace directory where logs and results will be collected.


### Default Behavior

If the `-w` option is not provided, the tool will:

1. Create a default workspace folder in the system's temporary directory.
2. Collect logs and results in this default workspace.
3. Compress the logs into a zip file.
4. Copy the zip file to the current working directory.

### More Detailed Output

Some environmental checks may gather more system information if the tool is launched with administrative privileges. For example, when listing processes that listen on a port, process names can only be obtained if the tool is launched with `sudo`. To do this just for the environment check, use the following command:

<!--pytest.mark.skip-->
```bash
sudo -E <venv>/bin/python -m pytriton env -w ...

# or if your virtual environment is activated

sudo -E $(which python3) -m pytriton env -w ...
```

If it is more convenient for you, you can also run the entire check with sudo, generating a single log with all the required information:

<!--pytest.mark.skip-->
```bash
sudo -E <venv>/bin/python -m pytriton check -w ...

# or if your virtual environment is activated

sudo -E $(which python3) -m pytriton check -w ...
```
