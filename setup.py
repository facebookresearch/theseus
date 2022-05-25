from pathlib import Path
import setuptools
import os
from torch.utils import cpp_extension as torch_cpp_ext


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
version = (root_dir / "version.txt").read_text().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

if "CUDA_HOME" in os.environ:
    ext_modules = [
        torch_cpp_ext.CUDAExtension(
            name="theseus.extlib.mat_mult", sources=["theseus/extlib/mat_mult.cu"]
        ),
        torch_cpp_ext.CUDAExtension(
            name="theseus.extlib.cusolver_lu_solver",
            sources=[
                "theseus/extlib/cusolver_lu_solver.cpp",
                "theseus/extlib/cusolver_sp_defs.cpp",
            ],
            libraries=["cusolver"],
        ),
    ]
else:
    ext_modules = []

setuptools.setup(
    name="theseus",
    version=version,
    author="Author",
    description="A library for differentiable nonlinear optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="anonymous url",
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
