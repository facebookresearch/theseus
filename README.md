![Theseus Logo](https://github.com/facebookresearch/theseus/blob/main/docs/source/img/theseus-color-horizontal.png)

[![CircleCI](https://circleci.com/gh/facebookresearch/theseus/tree/main.svg?style=svg)](https://circleci.com/gh/facebookresearch/theseus/tree/main)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/habitat-sim/blob/main/LICENSE)
[![python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)](https://www.python.org/downloads/release/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-green?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg)](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md)


Theseus is a library for differentiable nonlinear optimization built on PyTorch to support constructing various problems in robotics and vision as end-to-end differentiable architectures.

The current focus is on nonlinear least squares with support for sparsity, batching, GPU, and backward modes for unrolling, truncated and implicit, and sampling based differentiation. This library is in beta with expected full release in mid 2022.

-----

## Getting Started
- Prerequisites
    - We *strongly* recommend you install `theseus` in a venv or conda environment.
    - Theseus requires `torch` installation. To install for your particular CPU/CUDA configuration, follow the instructions in the PyTorch [website](https://pytorch.org/get-started/locally/).
    - Theseus also requires [`suitesparse`](https://people.engr.tamu.edu/davis/suitesparse.html), which you can install via:
        - `sudo apt-get install libsuitesparse-dev` (Ubuntu).
        - `conda install -c conda-forge suitesparse` (Mac).
    
- Installing
    ```bash
    git clone https://github.com/facebookresearch/theseus.git && cd theseus
    pip install -e .
    ```
    If you are interested in contributing to `theseus`, instead install
    ```bash
    pip install -e ".[dev]"
    ```
    and follow the more detailed instructions in [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md).
- Running unit tests
    ```bash
    pytest theseus
    ```
  By default, unit tests include tests for our CUDA extensions. You can add the option `-m "not cudaext"` to 
  skip them when installing without CUDA support.
- See [tutorials](https://github.com/facebookresearch/theseus/blob/main/tutorials/) and [examples](https://github.com/facebookresearch/theseus/blob/main/examples/) to learn about the API and usage.


## Additional Information

- Use [Github issues](https://github.com/facebookresearch/theseus/issues/new/choose) for questions, suggestions, and bugs.
- See [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md) if interested in helping out.


## License

Theseus is MIT licensed. See the [LICENSE](https://github.com/facebookresearch/theseus/blob/main/LICENSE) for details.
