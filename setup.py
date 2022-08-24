#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import setuptools
import sys

try:
    import torch
    from torch.utils import cpp_extension as torch_cpp_ext
except ModuleNotFoundError:
    print("Theseus installation requires torch.")
    sys.exit(1)


def parse_requirements_file(path):
    with open(path) as f:
        reqs = []
        for line in f:
            line = line.strip()
            reqs.append(line.split("==")[0])
    return reqs


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")
root_dir = Path(__file__).parent

with open(Path("theseus") / "__init__.py", "r") as f:
    for line in f:
        if "__version__" in line:
            version = line.split("__version__ = ")[1].rstrip().strip('"')

with open("README.md", "r") as fh:
    long_description = fh.read()

if torch.cuda.is_available():
    ext_modules = [
        torch_cpp_ext.CUDAExtension(
            name="theseus.extlib.mat_mult",
            sources=[str(root_dir / "theseus" / "extlib" / "mat_mult.cu")],
        ),
        torch_cpp_ext.CUDAExtension(
            name="theseus.extlib.cusolver_lu_solver",
            sources=[
                str(root_dir / "theseus" / "extlib" / "cusolver_lu_solver.cpp"),
                str(root_dir / "theseus" / "extlib" / "cusolver_sp_defs.cpp"),
            ],
            include_dirs=[str(root_dir)],
            libraries=["cusolver"],
        ),
    ]
else:
    print("No CUDA support found. CUDA extensions won't be installed.")
    ext_modules = []

setuptools.setup(
    name="theseus-ai",
    version=version,
    author="Meta Research",
    description="A library for differentiable nonlinear optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/theseus",
    keywords="differentiable optimization, nonlinear least squares, factor graphs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=reqs_main,
    extras_require={"dev": reqs_main + reqs_dev},
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    ext_modules=ext_modules,
)
