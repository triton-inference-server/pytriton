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
"""Wrapper for test of network timeouts with pytest"""

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def main():
    import argparse
    import os
    import subprocess
    import sys
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="activate verbose mode")
    args = parser.parse_args()
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    rel_path = os.path.relpath(script_dir, cwd)
    if args.verbose:
        test_command = [
            "pytest",
            "-v",
            "--log-cli-level=DEBUG",
            "--log-cli-format='%(asctime)s [%(levelname)s] [%(process)d:%(thread)d] %(message)s'",
            "--timeout=180",
            os.path.join(rel_path, "test_pytest.py"),
        ]
    else:
        test_command = ["pytest", "--timeout=180", os.path.join(rel_path, "test_pytest.py")]
    print("Test command:", test_command)  # noqa: T201 # pylint: disable=print-statement
    test_process = subprocess.Popen(test_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Define a function to read the output from a pipe and print it
    def read_output(pipe):
        for line in iter(pipe.readline, b""):
            print(line.decode(), end="")  # noqa: T201 # pylint: disable=print-statement

    stdout_thread = threading.Thread(target=read_output, args=(test_process.stdout,))
    stderr_thread = threading.Thread(target=read_output, args=(test_process.stderr,))

    # Start the threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the threads to finish
    stdout_thread.join()
    stderr_thread.join()

    test_process.wait()
    sys.exit(test_process.returncode)


if __name__ == "__main__":
    main()
