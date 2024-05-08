#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Script that match the current GLIBC version with the manylinux options."""

import argparse
import logging
import pathlib
import subprocess
from typing import Set, Union

from packaging.version import parse

LOGGER = logging.getLogger("manylinux_version")

# Policy list for auditwheel: https://github.com/pypa/auditwheel/blob/main/src/auditwheel/policy/manylinux-policy.json
MANYLINUX_GLIBC_VERSIONS = ["2.5", "2.12", "2.17", "2.24", "2.27", "2.28", "2.31", "2.34", "2.35"]


def _glibc_library_version(file_path: Union[pathlib.Path, str]) -> Set[str]:
    LOGGER.info(f"Collecting GLIBC version for {file_path}")
    cmd = f"objdump -T {file_path} | grep GLIBC_ | sed 's/.*GLIBC_\\([.0-9]*\\).*/\\1/g' | sort -u"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = proc.communicate()
    if stdout == b"\x01\n":  # WAR for unrecognized character
        LOGGER.info("GLIBC version not found.")
        return set()

    # Convert output and strip string
    output = stdout.decode("utf-8").strip()
    if not output:
        LOGGER.info("GLIBC version not found.")
        return set()

    # Split versions string to list and update set
    versions = set(output.splitlines())
    LOGGER.info(f"Collected versions: {versions}")
    return versions


def glibc_to_manylinux_version():
    """Find the minimal required GLIBC version supported by MANYLINUX version.

    The script parse the libraries and binaries located in --triton-path catalog.
    """
    argparser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    argparser.add_argument("--triton-path", type=pathlib.Path, required=True)

    args = argparser.parse_args()

    glibc_versions = set()
    for file_path in args.triton_path.glob("**/*"):
        # Filter unsupported files for
        if file_path.is_dir() or (file_path.suffixes and ".so" not in file_path.suffixes):
            continue

        file_glibc_versions = _glibc_library_version(file_path)
        glibc_versions.update(file_glibc_versions)

    LOGGER.info(f"Collected GLIBC versions {glibc_versions}")

    # Convert GLIBC and Manylinux version to semver supported comparison
    glibc_version = max(map(parse, glibc_versions))
    manylinux_glibc_versions = map(parse, MANYLINUX_GLIBC_VERSIONS)

    # Obtain the smallest compatible version based version obtained from libraries
    manylinux_glibc = min(filter(lambda i: i >= glibc_version, manylinux_glibc_versions))

    # Convert to wheel name format
    manylinux_version = str(manylinux_glibc).replace(".", "_")

    LOGGER.info(f"Selected manylinux versions {manylinux_version}")
    # Print output to collect value in shell
    print(manylinux_version)  # noqa: T201


if __name__ == "__main__":
    glibc_to_manylinux_version()
