# Theseus

A library for differentiable nonlinear optimization built on PyTorch to support constructing various problems in robotics and vision as end-to-end differentiable architectures.

The current focus is on nonlinear least squares with support for sparsity, batching, GPU, and backward modes for unrolling, truncated and implicit, and sampling based differentiation.


## Getting Started
- Prerequisites
    - We *strongly* recommend you install `theseus` in a venv or conda environment.
    - Theseus requires `torch` installation. To install for your particular CPU/CUDA configuration, follow the instructions in the PyTorch [website](https://pytorch.org/get-started/locally/).
    - Theseus also requires [`suitesparse`](https://people.engr.tamu.edu/davis/suitesparse.html), which you can install via:
        - `sudo apt-get install libsuitesparse-dev` (Ubuntu).
        - `conda install -c conda-forge suitesparse` (Mac).
    
- Installing
    ```bash
    cd theseus
    pip install -e .
    ```

- Running unit tests
    ```bash
    pytest theseus
    ```

## NeurIPS'22 Experiments
See the `neurips22_scripts` folder for scripts that run the experiments described in the paper.
The python scripts with the learning loops can be found under the `examples` folder. 