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
"""Script with hook for replacing ../<anything> markdown links in docs to git repo."""
import logging
import os
import pathlib
import re


def on_page_markdown(markdown: str, config, **kwargs):
    """Called on markdown content."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t -  %(name)s: %(message)s")
    logger = logging.getLogger("scripts.rewrite_links_to_repo")

    ref = _get_current_ref()

    def _replace(_md_path):
        repo_url = config["repo_url"]
        view_uri_template = config["view_uri_template"]
        path = pathlib.Path("docs") / _md_path
        path = path.resolve().relative_to(pathlib.Path.cwd())
        full_url = f"{repo_url}/{view_uri_template.format(ref=ref, path=path)}"
        return full_url

    for md_path in _extract_external_link(markdown):
        logger.info(f"replacing {md_path} -> {_replace(md_path)}")
        markdown = markdown.replace(md_path, _replace(md_path))

    return markdown


def _get_current_ref():
    ref = os.environ.get("CI_COMMIT_REF_NAME", None)
    if ref is None:
        import git

        try:
            repo = git.Repo(".")
            ref = repo.active_branch.name
        except (git.InvalidGitRepositoryError, TypeError):
            # TypeError thrown on non detached head - no active branch
            ref = "main"
    return ref


def _extract_external_link(markdown):
    external_paths = [path_with_brackets[1:-1] for path_with_brackets in re.findall(r"\(\.\.\/.*\)", markdown)]
    return list(set(external_paths))
