<!--
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

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

# Contributing

Contributions are welcome, and they are much appreciated! Every little
helps, and we will always give credit.

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/triton-inference-server/pytriton/issues](https://github.com/triton-inference-server/pytriton/issues).

When reporting a bug, please include the following information:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Browse through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

The PyTriton could always use more documentation, whether as part of
the official PyTriton docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/triton-inference-server/pytriton/issues](https://github.com/triton-inference-server/pytriton/issues).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible to make it easier to implement.

## Sign your Work

We require that all contributors "sign-off" on their commits. This certifies that
the contribution is your original work, or you have the rights to submit it under
the same license or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit, simply use the `--signoff` (or `-s`) option when committing your changes:

```shell
$ git commit -s -m "Add a cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

By doing this, you certify the following:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

## Get Started!

### Local Development

Ready to contribute? Here's how to set up the `PyTriton` for local development.

1. Fork the `PyTriton` repo on GitHub.
2. Clone your fork locally:

```shell
$ git clone git@github.com:your_name_here/pytriton.git
```

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, here's how you set up your fork for local development:

```shell
$ mkvirtualenv pytriton
$ cd pytriton/
```

If you do not use the virtualenvwrapper package, you can initialize a virtual environment using the pure Python command:

```shell
$ python -m venv pytriton
$ cd pytriton/
$ source bin/activate
```

Once the virtualenv is activated, install the development dependencies:

```shell
$ make install-dev
```

4. Extract Triton Server to your environment so you can debug PyTriton while serving some models on Triton:

```shell
$ make extract-triton
```

5. Install pre-commit hooks:

```shell
$ pre-commit install
```

6. Create a branch for local development:

```shell
$ git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

7. When you're done making changes, check that your changes pass linters and the
   tests, including testing other Python versions with tox:

```shell
$ make lint  # will run, among others, flake8 and pytype linters
$ make test  # will run a test on your current virtualenv
```

  To run a subset of tests:

```shell
$ pytest tests.test_subset
```

8. Commit your changes and push your branch to GitHub:

```shell
$ git add .
$ git commit -s -m "Your detailed description of your changes."
$ git push origin name-of-your-bugfix-or-feature
```

9. Submit a pull request through the GitHub website.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, you should update the docs. Put your new functionality into a function with a docstring and add the feature to the list in README.md.


## Documentation

Add/update docstrings as defined in [Google Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).

## Contributor License Agreement (CLA)

PyTriton requires that all contributors (or their corporate entity) send
a signed copy of the [Contributor License
Agreement](https://github.com/NVIDIA/triton-inference-server/blob/master/Triton-CCLA-v1.pdf)
to triton-cla@nvidia.com.

*NOTE*: Contributors with no company affiliation can fill `N/A` in the
`Corporation Name` and `Corporation Address` fields.
