#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import os
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


def get_baspacho_info(baspacho_root_dir, has_cuda):
    baspacho_build_dir = Path(baspacho_root_dir) / "build"
    include_dirs = [
        str(baspacho_root_dir),
        str(baspacho_build_dir / "_deps" / "eigen-src"),
    ]
    library_dirs = [
        str(baspacho_build_dir / "baspacho" / "baspacho"),
        str(baspacho_build_dir / "_deps" / "dispenso-build" / "dispenso"),
    ]
    libraries = ["BaSpaCho", "dispenso"]
    if has_cuda:
        libraries.append("cusolver")
        libraries.append("cublas")
    return include_dirs, library_dirs, libraries


def maybe_create_baspacho_extension(has_cuda):
    baspacho_root_dir = os.environ.get("BASPACHO_ROOT_DIR", None)
    if baspacho_root_dir is None:
        print("No BaSpaCho dir given, so extension will not be installed.")
        return None
    ext_cls = torch_cpp_ext.CUDAExtension if has_cuda else torch_cpp_ext.CppExtension
    sources = ["theseus/extlib/baspacho_solver.cpp"]
    define_macros = [("NO_BASPACHO_CHECKS", "1")]
    if has_cuda:
        sources.append("theseus/extlib/baspacho_solver_cuda.cu")
        define_macros.append(("THESEUS_HAVE_CUDA", "1"))
        extra_compile_args = {"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]}
    else:
        extra_compile_args = ["-std=c++17"]
    include_dirs, library_dirs, libraries = get_baspacho_info(
        baspacho_root_dir, has_cuda
    )
    return ext_cls(
        name="theseus.extlib.baspacho_solver",
        sources=sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    )


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")
root_dir = Path(__file__).parent

with open(Path("theseus") / "__init__.py", "r") as f:
    for line in f:
        if "__version__" in line:
            version = line.split("__version__ = ")[1].rstrip().strip('"')

with open("README.md", "r") as fh:
    long_description = fh.read()

# Add C++ and CUDA extensions
cuda_is_available = torch.cuda.is_available()
if cuda_is_available:
    ext_modules = [
        # reference: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
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
baspacho_extension = maybe_create_baspacho_extension(cuda_is_available)
if baspacho_extension is not None:
    ext_modules.append(baspacho_extension)

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
