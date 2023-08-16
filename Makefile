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
.PHONY: clean clean-build clean-tritonserver clean-pyc clean-docs clean-test docs lint test coverage release dist extract-triton install install-dev help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
PIP_INSTALL := pip install --extra-index-url https://pypi.ngc.nvidia.com
TRITONSERVER_IMAGE_VERSION = 23.07
TRITONSERVER_IMAGE_NAME = nvcr.io/nvidia/tritonserver:$(TRITONSERVER_IMAGE_VERSION)-pyt-python-py3
TRITONSERVER_OUTPUT_DIR = pytriton/tritonserver
# to set PLATFORM from outside, use: make PLATFORM=linux/aarch64;
# correct values are: linux/x86_64 (default), linux/aarch64
PLATFORM=linux/x86_64

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-tritonserver clean-docs ## remove all build, tritonserver, test, docs, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-tritonserver:
	rm -fr pytriton/tritonserver

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-docs: ## remove test and coverage artifacts
	rm -rf site

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .pytype/

docs: clean-docs ## generate site
	cp CHANGELOG.md docs
	cp CONTRIBUTING.md docs
	cp LICENSE docs/LICENSE.md
	cp examples/README.md docs/examples.md
	mkdocs build --clean

docs-serve: docs
	mkdocs serve

lint: ## check style with pre-commit and pytype
	tox -e pytype,pre-commit --develop

test: ## run tests on every Python version with tox
	tox --develop --skip-missing-interpreters

coverage: ## check code coverage quickly with the default Python
	coverage run --source pytriton -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

dist: clean extract-triton ## builds source and wheel package
	python3 -m build .
	find ./dist -iname *-linux*.whl -type f -exec bash ./scripts/add_libs_to_wheel.sh $(TRITONSERVER_IMAGE_NAME) $(TRITONSERVER_OUTPUT_DIR) {} ${PLATFORM} \;
	find ./dist -iname *-linux*.whl -type f -delete
	ls -lh dist
	twine check dist/*

extract-triton:
	# changing dst path, change also in clean-build and pyproject.toml
	bash ./scripts/extract_triton.sh $(TRITONSERVER_IMAGE_NAME) $(TRITONSERVER_OUTPUT_DIR) $(PLATFORM)

install: clean extract-triton ## install the package to the active Python's site-packages
	$(PIP_INSTALL) --upgrade pip
	$(PIP_INSTALL) .

install-dev: clean-build clean-pyc
	$(PIP_INSTALL) --upgrade pip
	$(PIP_INSTALL) -e .[dev]
