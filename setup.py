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

    # This is a hack to get CUDAExtension to compile with
    #   -ltorch_cuda
    # instead of split
    #   -ltorch_cuda_cu -ltorch_cuda_cpp
    torch_cpp_ext.BUILD_SPLIT_CUDA = False

    if hasattr(torch_cpp_ext, "CUDA_GCC_VERSIONS"):
        # hack to be able to compile with gcc-8.4.0
        torch_cpp_ext.CUDA_GCC_VERSIONS["10.2"] = (
            torch_cpp_ext.MINIMUM_GCC_VERSION,
            (8, 4, 99),
        )

    torch_version = torch.__version__.split(".")
    torch_geq_113 = (
        int(torch_version[0]) > 1
        or int(torch_version[0]) == 1
        and int(torch_version[1]) >= 13
    )
except ModuleNotFoundError:
    print("Theseus installation requires torch.")
    sys.exit(1)


def parse_requirements_file(path):
    with open(path) as f:
        reqs = []
        for line in f:
            if "functorch" in line and torch_geq_113:
                # Don't install functorch 0.2.1 if torch 1.13 already
                # installed
                continue
            line = line.strip()
            reqs.append(line)
    return reqs


def get_baspacho_info(baspacho_root_dir, has_cuda):
    baspacho_build_dir = Path(baspacho_root_dir) / "build"
    include_dirs = [
        str(baspacho_root_dir),
        str(baspacho_build_dir / "_deps" / "eigen-src"),
    ]
    library_dirs = [
        str(baspacho_build_dir / "baspacho" / "baspacho"),
    ]
    libraries = ["BaSpaCho"]
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
    include_dirs, library_dirs, libraries = get_baspacho_info(
        baspacho_root_dir, has_cuda
    )
    return ext_cls(
        name="theseus.extlib.baspacho_solver",
        sources=sources,
        define_macros=define_macros,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    )


reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")
root_dir = Path(__file__).parent

is_nightly = False
nightly_date_str = os.environ.get("THESEUS_NIGHTLY", None)
if nightly_date_str is not None:
    from datetime import date, datetime

    nightly_date = datetime.strptime(nightly_date_str, "%Y.%m.%d").date()
    assert nightly_date == date.today(), (
        f"THESEUS_NIGHTLY must be set to today's date in format %Y.%-m.%-d (stripped) "
        f"but got {nightly_date_str}."
    )
    print(f"Building nightly with date {nightly_date_str}")
    is_nightly = True

if is_nightly:
    version = nightly_date_str
else:
    with open(Path("theseus") / "__init__.py", "r") as f:
        for line in f:
            if "__version__" in line:
                version = line.split("__version__ = ")[1].rstrip().strip('"')

with open("README.md", "r") as fh:
    long_description = fh.read()

# Add C++ and CUDA extensions
compile_cuda_flag = os.environ.get("THESEUS_FORCE_CUDA")
compile_cuda_support = (
    torch.cuda.is_available()
    if (compile_cuda_flag is None)
    else (compile_cuda_flag not in {"", "0", "False"})
)
cuda_detection_info = (
    "detected" if compile_cuda_flag is None else "forced by THESEUS_FORCE_CUDA env var"
)
print(f"Theseus CUDA support: {compile_cuda_support} ({cuda_detection_info})")

if compile_cuda_support:
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
baspacho_extension = maybe_create_baspacho_extension(compile_cuda_support)
if baspacho_extension is not None:
    ext_modules.append(baspacho_extension)

excluded_packages = []
package_name = "theseus-ai-nightly" if is_nightly else "theseus-ai"
if not os.environ.get("INCLUDE_THESEUS_LABS") and not is_nightly:
    excluded_packages.append("theseus.labs")
    print("Excluding theseus.labs")
setuptools.setup(
    name=package_name,
    version=version,
    author="Meta Research",
    description="A library for differentiable nonlinear optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/theseus",
    keywords="differentiable optimization, nonlinear least squares, factor graphs",
    packages=setuptools.find_packages(exclude=tuple(excluded_packages)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=reqs_main,
    extras_require={"dev": reqs_main + reqs_dev},
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    ext_modules=ext_modules,
)
