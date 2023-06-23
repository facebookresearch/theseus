#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import setuptools
from pathlib import Path

lie_path = Path("torchlie")
with open(lie_path / "__init__.py", "r") as f:
    for line in f:
        if "__version__" in line:
            version = line.split("__version__ = ")[1].rstrip().strip('"')

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="torchlie",
    version=version,
    author="Meta Research",
    description="Torch extension for differentiable Lie groups.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/theseus/lie",
    keywords="lie groups, differentiable optimization",
    packages=["torchlie", "torchlie.functional"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
