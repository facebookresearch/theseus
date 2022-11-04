![](https://raw.githubusercontent.com/facebookresearch/theseus/main/docs/source/img/theseus-color-horizontal.png)

<p align="center">
    <!-- CI -->
    <a href="https://circleci.com/gh/facebookresearch/theseus/tree/main">
        <img src="https://circleci.com/gh/facebookresearch/theseus/tree/main.svg?style=svg" alt="CircleCI" height="20">
    </a>
    <!-- License -->
    <a href="https://github.com/facebookresearch/theseus/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" height="20">
    </a>
    <!-- pypi -->
    <a href="https://pypi.org/project/theseus-ai/">
        <img src="https://img.shields.io/pypi/v/theseus-ai" alt="pypi"
        heigh="20">
    <!-- Downloads counter -->
    <a href="https://pypi.org/project/theseus-ai/">
        <img src="https://pepy.tech/badge/theseus-ai" alt="PyPi Downloads" height="20">
    </a>
    <!-- Python -->
    <a href="https://www.python.org/downloads/release/">
        <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg" alt="Python" height="20">
    </a>
    <!-- Pre-commit -->
    <a href="https://github.com/pre-commit/pre-commit">
        <img src="https://img.shields.io/badge/pre--commit-enabled-green?logo=pre-commit&logoColor=white" alt="pre-commit" height="20">
    </a>
    <!-- Black -->
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="black" height="20">
    </a>
    <!-- PRs -->
    <a href="https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md">
        <img src="https://img.shields.io/badge/PRs-welcome-green.svg" alt="PRs" height="20">
    </a>
</p>

<p align="center">
    <i>A library for differentiable nonlinear optimization</i>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2207.09442">Paper</a> •
    <a href="https://ai.facebook.com/blog/theseus-a-library-for-encoding-domain-knowledge-in-end-to-end-ai-models/">Blog</a> •
    <a href="https://sites.google.com/view/theseus-ai/">Webpage</a> •
    <a href="https://github.com/facebookresearch/theseus/tree/main/tutorials">Tutorials</a> •
    <a href="https://theseus-ai.readthedocs.io/">Docs</a>
</p>

Theseus is an efficient application-agnostic library for building custom nonlinear optimization layers in PyTorch to support constructing various problems in robotics and vision as end-to-end differentiable architectures.

![](https://raw.githubusercontent.com/facebookresearch/theseus/main/docs/source/img/theseuslayer.png)

Differentiable nonlinear optimization provides a general scheme to encode inductive priors, as the objective function can be partly parameterized by neural models and partly with expert domain-specific differentiable models. The ability to compute gradients end-to-end is retained by differentiating through the optimizer which allows neural models to train on the final task loss, while also taking advantage of priors captured by the optimizer.

-----

## Current Features

### Application agnostic interface
Our implementation provides an easy to use interface to build custom optimization layers and plug them into any neural architecture. Following differentiable features are currently available:
- [Second-order nonlinear optimizers](https://github.com/facebookresearch/theseus/tree/main/theseus/optimizer/nonlinear)
    - Gauss-Newton, Levenberg–Marquardt
- [Linear solvers](https://github.com/facebookresearch/theseus/tree/main/theseus/optimizer/linear)
    - Dense: Cholesky, LU; Sparse: CHOLMOD, LU (GPU-only), [BaSpaCho](https://github.com/facebookresearch/baspacho)
- [Commonly used costs](https://github.com/facebookresearch/theseus/tree/main/theseus/embodied), [AutoDiffCostFunction](https://github.com/facebookresearch/theseus/blob/main/theseus/core/cost_function.py), [RobustCostFunction](https://github.com/facebookresearch/theseus/blob/main/theseus/core/robust_cost_function.py)
- [Lie groups](https://github.com/facebookresearch/theseus/tree/main/theseus/geometry)
- [Robot kinematics](https://github.com/facebookresearch/theseus/blob/main/theseus/embodied/kinematics/kinematics_model.py)

### Efficiency based design
We support several features that improve computation times and memory consumption:
- [Sparse linear solvers](https://github.com/facebookresearch/theseus/tree/main/theseus/optimizer/linear)
- Batching and GPU acceleration
- [Automatic vectorization](https://github.com/facebookresearch/theseus/blob/main/theseus/core/vectorizer.py)
- [Backward modes](https://github.com/facebookresearch/theseus/blob/main/theseus/optimizer/nonlinear/nonlinear_optimizer.py)
    - Implicit, Truncated, Direct Loss Minimization ([DLM](https://github.com/facebookresearch/theseus/blob/main/theseus/theseus_layer.py)), Sampling ([LEO](https://github.com/facebookresearch/theseus/blob/main/examples/state_estimation_2d.py))


## Getting Started

### Prerequisites
- We *strongly* recommend you install Theseus in a venv or conda environment with Python 3.7-3.9.
- Theseus requires `torch` installation. To install for your particular CPU/CUDA configuration, follow the instructions in the PyTorch [website](https://pytorch.org/get-started/locally/).
- For GPU support, Theseus requires [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) to compile custom CUDA operations. Make sure it matches the version used to compile pytorch with `nvcc --version`. If not, install it and ensure its location is on your system's `$PATH` variable.
- Theseus also requires [`suitesparse`](https://people.engr.tamu.edu/davis/suitesparse.html), which you can install via:
    - `sudo apt-get install libsuitesparse-dev` (Ubuntu).
    - `conda install -c conda-forge suitesparse` (Mac).
    
### Installing

- **pypi**
    ```bash
    pip install theseus-ai
    ```
    We currently provide wheels with our CUDA extensions compiled using CUDA 10.2 and Python 3.9.
    For other CUDA versions, consider installing from source or using our 
    [build script](https://github.com/facebookresearch/theseus/blob/main/build_scripts/build_wheel.sh).

- #### **From source**
    The simplest way to install Theseus from source is by running the following (see further below to also include BaSpaCho)
    ```bash
    git clone https://github.com/facebookresearch/theseus.git && cd theseus
    pip install -e .
    ```
    If you are interested in contributing to Theseus, instead install
    ```bash
    pip install -e ".[dev]"
    ```
    and follow the more detailed instructions in [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md).

- **Installing BaSpaCho extensions from source**

    By default, installing from source doesn't include our BaSpaCho sparse solver extension. For this, follow these steps:

    1. Compile BaSpaCho from source following instructions [here](https://github.com/facebookresearch/baspacho). We recommend using flags `-DBLA_STATIC=ON -DBUILD_SHARED_LIBS=OFF`.
    2. Run
        
        ```bash
        git clone https://github.com/facebookresearch/theseus.git && cd theseus
        BASPACHO_ROOT_DIR=<path/to/root/baspacho/dir> pip install -e .
        ```
        
        where the BaSpaCho root dir must have the binaries in the subdirectory `build`.

### Running unit tests (requires `dev` installation)
```bash
python -m pytest tests
```
By default, unit tests include tests for our CUDA extensions. You can add the option `-m "not cudaext"`
to skip them when installing without CUDA support. Additionally, the tests for sparse solver BaSpaCho are automatically 
skipped when its extlib is not compiled.


## Examples

[Simple example](https://github.com/facebookresearch/theseus/blob/main/examples/simple_example.py). This example is fitting the curve $y$ to a dataset of $N$ observations $(x,y) \sim D$. This is modeled as an `Objective` with a single `CostFunction` that computes the residual $y - v e^x$. The `Objective` and the `GaussNewton` optimizer are encapsulated into a `TheseusLayer`. With `Adam` and MSE loss, $x$ is learned by differentiating through the `TheseusLayer`.

```python
import torch
import theseus as th

x_true, y_true, v_true = read_data() # shapes (1, N), (1, N), (1, 1)
x = th.Variable(torch.randn_like(x_true), name="x")
y = th.Variable(y_true, name="y")
v = th.Vector(1, name="v") # a manifold subclass of Variable for optim_vars

def error_fn(optim_vars, aux_vars): # returns y - v * exp(x)
    x, y = aux_vars
    return y.tensor - optim_vars[0].tensor * torch.exp(x.tensor)

objective = th.Objective()
cost_function = th.AutoDiffCostFunction(
    [v], error_fn, y_true.shape[1], aux_vars=[x, y],
    cost_weight=th.ScaleCostWeight(1.0))
objective.add(cost_function)
layer = th.TheseusLayer(th.GaussNewton(objective, max_iterations=10))

phi = torch.nn.Parameter(x_true + 0.1 * torch.ones_like(x_true))
outer_optimizer = torch.optim.Adam([phi], lr=0.001)
for epoch in range(10):
    solution, info = layer.forward(
        input_tensors={"x": phi.clone(), "v": torch.ones(1, 1)},
        optimizer_kwargs={"backward_mode": "implicit"})
    outer_loss = torch.nn.functional.mse_loss(solution["v"], v_true)
    outer_loss.backward()
    outer_optimizer.step()
```

See [tutorials](https://github.com/facebookresearch/theseus/blob/main/tutorials/), and robotics and vision [examples](https://github.com/facebookresearch/theseus/tree/main/examples) to learn about the API and usage.


## Citing Theseus

If you use Theseus in your work, please cite the [paper](https://arxiv.org/abs/2207.09442) with the BibTeX below.

```bibtex
@article{pineda2022theseus,
  title   = {{Theseus: A Library for Differentiable Nonlinear Optimization}},
  author  = {Luis Pineda and Taosha Fan and Maurizio Monge and Shobha Venkataraman and Paloma Sodhi and Ricky TQ Chen and Joseph Ortiz and Daniel DeTone and Austin Wang and Stuart Anderson and Jing Dong and Brandon Amos and Mustafa Mukadam},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```


## License

Theseus is MIT licensed. See the [LICENSE](https://github.com/facebookresearch/theseus/blob/main/LICENSE) for details.


## Additional Information

- Join the community on [Github Discussions](https://github.com/facebookresearch/theseus/discussions) for questions and sugesstions.
- Use [Github Issues](https://github.com/facebookresearch/theseus/issues/new/choose) for bugs and features.
- See [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md) if interested in helping out.

Theseus is made possible by the following contributors:

<a href="https://github.com/facebookresearch/theseus/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=facebookresearch/theseus" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
