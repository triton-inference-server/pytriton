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
"""Pytriton check module."""

import logging
import os
import pathlib
import shutil
import tempfile
from typing import Optional

import typer
from typing_extensions import Annotated

from pytriton.check.add_sub import add_sub_example, add_sub_example_thread
from pytriton.check.env_checks import env_checks

warning_message = """
+---------------------------------------------------------------+
|                             WARNING                           |
+---------------------------------------------------------------+
| Command may collect sensitive information, please review the  |
| log and the ZIP before sharing.                               |
+---------------------------------------------------------------+
"""


app = typer.Typer(help="Pytriton check tool.\n\nThis tool is used to check the environment and run examples.")


class CheckEnvironment:
    """Check environment class.

    Args:
        workspace_path: Path to workspace
        name: Name of the sub_workspace
        zip_results: Flag if results should be zipped
        check_workspace_exist: Flag if workspace should be checked if exists
    """

    def __init__(
        self,
        workspace_path: Optional[pathlib.Path],
        name: str,
        zip_results: bool = True,
        check_workspace_exist: bool = True,
    ):
        """Initialize class."""
        self.name = name
        self._zip_results = zip_results
        self._temp_workspace = None

        self.logger = logging.getLogger(name)
        if check_workspace_exist and workspace_path is not None and workspace_path.exists():
            self.logger.error(f"Workspace path {workspace_path} already exists")
            raise typer.Exit(code=1)
        if workspace_path is None:
            self._temp_workspace = tempfile.TemporaryDirectory(prefix="pytriton_workspace_")
            workspace_path = pathlib.Path(self._temp_workspace.name)
        else:
            workspace_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
        self.logger.addHandler(logging.FileHandler(workspace_path / (name + "_log.txt")))
        self.workspace_path = workspace_path
        self.sub_workspace = workspace_path / name

    def __enter__(self):
        """Enter method."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method zips results if required."""
        self.zip_results()

    def zip_results(self):
        """Zip results."""
        if self._zip_results:
            if self.workspace_path.exists():
                if self._temp_workspace is not None:
                    output_file_base = pathlib.Path(os.getcwd()) / self.workspace_path.name
                else:
                    output_file_base = self.workspace_path
                self.logger.info(f"Zipping {self.workspace_path} to {output_file_base}.zip")
                shutil.make_archive(str(output_file_base.resolve()), "zip", str(self.workspace_path.resolve()))
            else:
                self.logger.error(f"Workspace path {self.workspace_path} does not exist")


@app.command("example-add-sub-script")
def example_add_sub_script(
    workspace: Annotated[Optional[pathlib.Path], typer.Option("--workspace", "-w")] = None,
    zip_results: Annotated[bool, typer.Option("--zip")] = True,
):
    """Run example using external script.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        zip_results: flag if output should be zipped
    """
    with CheckEnvironment(workspace, "example_add_sub_script", zip_results) as ce:
        try:
            add_sub_example_thread(ce.sub_workspace, ce.logger)
        except Exception as e:
            ce.logger.error(f"Error occurred in command: {e}")


@app.command("example-add-sub")
def example_add_sub(
    workspace: Annotated[Optional[pathlib.Path], typer.Option("--workspace", "-w")] = None,
    zip_results: Annotated[bool, typer.Option("--zip")] = True,
):
    """Run example.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        zip_results: flag if output should be zipped
    """
    with CheckEnvironment(workspace, "example_add_sub", zip_results) as ce:
        try:
            add_sub_example(ce.sub_workspace, ce.logger)
        except Exception as e:
            ce.logger.error(f"Error occurred in command: {e}")


@app.command("examples")
def examples(
    workspace: Annotated[Optional[pathlib.Path], typer.Option("--workspace", "-w")] = None,
    zip_results: Annotated[bool, typer.Option("--zip")] = True,
):
    """Run example in the same process.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        zip_results: flag if output should be zipped
    """
    with CheckEnvironment(workspace, "example_add_sub", zip_results) as ce:
        try:
            add_sub_example(ce.sub_workspace, ce.logger)
        except Exception as e:
            ce.logger.error(f"Error occurred in command: {e}")

    with CheckEnvironment(workspace, "example_add_sub_script", zip_results, check_workspace_exist=False) as ce:
        try:
            add_sub_example_thread(ce.sub_workspace, ce.logger)
        except Exception as e:
            ce.logger.error(f"Error occurred in command: {e}")


@app.command("env")
def env_check(
    workspace: Annotated[Optional[pathlib.Path], typer.Option("--workspace", "-w")] = None,
    zip_results: Annotated[bool, typer.Option("--zip")] = True,
):
    """Run all environment checks.

    It may collect sensitive system information in the log. Please review the log before sharing.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        zip_results: flag if output should be zipped
    """
    with CheckEnvironment(workspace, "env_checks", zip_results) as ce:
        try:
            env_checks(ce.logger)
        except Exception as e:
            ce.logger.error(f"Error occurred in command: {e}")


@app.command("check")
def check(
    workspace: Annotated[Optional[pathlib.Path], typer.Option("--workspace", "-w")] = None,
    zip_results: Annotated[bool, typer.Option("--zip")] = True,
):
    """Run all checks.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        zip_results: flag if output should be zipped
    """
    with CheckEnvironment(workspace, "all_checks", zip_results) as ce:
        try:
            ce.logger.info("Running all common checks")
            env_check(ce.workspace_path / "env", False)
            examples(ce.workspace_path / "examples", False)
        except Exception as e:
            ce.logger.error(f"Error occurred in command: {e}")


@app.callback(invoke_without_command=True)
def default_command(ctx: typer.Context):
    """Default command."""
    if ctx.invoked_subcommand is None:
        check()


def main():
    """Main function."""
    logger = logging.getLogger("PyTriton-Check")
    try:
        logger.warning(warning_message)
        app()
    finally:
        logger.warning(warning_message)


if __name__ == "__main__":
    main()
