# torchlie

<p align="center">
    <!-- License -->
    <a href="https://github.com/facebookresearch/theseus/blob/main/torchlie/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" height="20">
    </a>
    <!-- pypi -->
    <a href="https://pypi.org/project/torchlie/">
        <img src="https://img.shields.io/pypi/v/torchlie" alt="pypi"
        heigh="20">
    <!-- Downloads counter -->
    <a href="https://pypi.org/project/torchlie/">
        <img src="https://pepy.tech/badge/torchlie" alt="PyPi Downloads" height="20">
    </a>
    <!-- Python -->
    <a href="https://www.python.org/downloads/release/">
        <img src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg" alt="Python" height="20">
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
    <i>A library for differentiable Lie groups</i>
</p>

-----

## Getting Started

### Prerequisites
- We *strongly* recommend you install torchlie in a venv or conda environment with Python 3.8-3.10.
- torchlie requires `torch` installation. To install for your particular CPU/CUDA configuration, follow the instructions in the PyTorch [website](https://pytorch.org/get-started/locally/).

### Installing

- **pypi**
    ```bash
    pip install torchlie
    ```

- #### **From source**
    The simplest way to install torchlie from source is by running the following
    ```bash
    git clone https://github.com/facebookresearch/theseus.git && cd theseus/torchlie
    pip install -e .
    ```
    If you are interested in contributing to torchlie, also install
    ```bash
    pip install -r ../requirements/dev.txt
    pre-commit install
    ```
    and follow the more detailed instructions in [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md).


## Example

The example below is also available as a [script](https://github.com/facebookresearch/theseus/blob/main/examples/torchlie_api.py).

```python
import torch

import torchlie as lie
import torchlie.functional as lieF

batch_size = 5

# ### Lie Tensor creation functions
g1 = lie.SE3.rand(batch_size, requires_grad=True)
print(f"Created SE3 tensor with shape {g1.shape}")
g2 = g1.clone()

# Identity element
i1 = lie.SO3.identity(2)
i2 = lie.SE3.identity(2)
print("SO3 identity", i1, i1.shape)
print("SE3 identity", i2, i2.shape)

# Indexing
g1_slice = g1[2:4]
assert g1_slice.shape == (2, 3, 4)
torch.testing.assert_close(g1_slice._t, g1._t[2:4])  # type: ignore
try:
    bad = g1[3, 2]
except NotImplementedError:
    print("INDEXING ERROR: Can only slice the first dimension for now.")

# ## Different constructors
g3_data = lieF.SO3.rand(5, requires_grad=True)  # this is a regular tensor with SO3 data

# Can create from a tensor as long as it's consistent with the desired ltype
g3 = lie.from_tensor(g3_data, lie.SO3)  # keeps grad history
assert g3.grad_fn is not None
try:
    x = lie.from_tensor(torch.zeros(1, 3, 3), lie.SO3)
except ValueError as e:
    print(f"ERROR: {e}")


def is_shared(t1, t2):  # utility to check if memory is shared
    return t1.storage().data_ptr() == t2.storage().data_ptr()


# # Let's check different copy vs no-copy options
# -- lie.SO3() lie.SE3()
g3_leaf = lie.SO3(g3_data)  # creates a leaf tensor and copies data
assert g3_leaf.grad_fn is None
assert not is_shared(g3_leaf, g3_data)

# -- lie.LieTensor() constructor is equivalent to lie.SO3()
g3_leaf_2 = lie.LieTensor(g3_data, lie.SO3)
assert g3_leaf_2.grad_fn is None
assert not is_shared(g3_leaf_2, g3_data)


# -- as_lietensor()
g4 = lie.as_lietensor(g3_data, lie.SO3)
assert is_shared(g3_data, g4)  # shares storage if possible
assert g4.grad_fn is not None  # result is not a leaf tensor
# Calling with a LieTensor returns the same tensor...
g5 = lie.as_lietensor(g3, lie.SO3)
assert g5 is g3
# ... unless dtype or device is different
g5_double = lie.as_lietensor(g3, lie.SO3, dtype=torch.double)
assert g5_double is not g3
assert not is_shared(g5_double, g3)

# -- cast()
g6 = lie.cast(g3_data, lie.SO3)  # alias for as_lietensor()
assert is_shared(g3_data, g6)

# -- LieTensor.new()
g7 = g3.new_lietensor(g3_data)
assert not is_shared(g3_data, g7)  # doesn't share storage
assert g7.grad_fn is None  # creates a leaf

# ### Lie operations
v = torch.randn(batch_size, 6)

# Exponential and logarithmic map
out1 = lie.SE3.exp(v)  # also lie.exp(v, g1.ltype)
print(f"Exp map returns a {type(out1)}.")
out2 = g1.log()  # also lie.log(g1)
print(f"Log map returns a {type(out2)}.")

# Inverse
out1 = g1.inv()  # also lie.inv(g1)

# Compose
# also lie.compose(g1, g2)
out1 = g1.compose(g2)  # type: ignore

# Differentiable jacobians
jacs, out = g1.jcompose(g2)  # type: ignore
print("Jacobians output is a 2-tuple.")
print("    First element is a list of jacobians, one per group argument.")
print(f"    For compose this means length {len(jacs)}.")
print("    The second element of the tuple is the result of the operation itself.")
print(f"    Which for compose is a {type(out).__name__}.")

# Other options:
#   * adj(), hat(), vee(), retract(), local(),
#   * Jacobians: jlog(), jinv(), jexp()

# ### Overriden operators
# Compose
out2 = g1 * g2
torch.testing.assert_close(out1, out2, check_dtype=True)

# Transfrom from (from local to world coordinate frame)
p = torch.randn(batch_size, 3)
pt1 = g1.transform_from(p)
pt2 = g1 @ p
torch.testing.assert_close(pt1, pt2)

# For convenience, we provide a context to drop all ltype checks, and operate
# on raw tensor data. However, keep in mind that this is prone to error.
# Here is one example of how this works.
with lie.as_euclidean():
    gg1 = torch.sin(g1)
# The above is the same as this next call, but the context might be more convenient
# if one is doing similar hacky stuff on several group objects.
gg2 = torch.sin(g1._t)
torch.testing.assert_close(gg1, gg2)
print("Success: We just did some ops that make no sense for SE3 tensors.")

# ### Lie tensors can also be used as leaf tensors for torch optimizers
g1 = lie.SE3.rand(1, requires_grad=True)
g2 = lie.SE3.rand(1)

opt = torch.optim.Adam([g1], lr=0.1)

for i in range(10):
    opt.zero_grad()
    d = g1.local(g2)
    loss = torch.sum(d**2)
    loss.backward()
    opt.step()
    print(f"Iter {i}. Loss: {loss.item(): .3f}")
```


## Citing torchlie

If you use torchlie in your work, please cite the [paper](https://arxiv.org/abs/2207.09442) with the BibTeX below.

```bibtex
@article{pineda2022theseus,
  title   = {{Theseus: A Library for Differentiable Nonlinear Optimization}},
  author  = {Luis Pineda and Taosha Fan and Maurizio Monge and Shobha Venkataraman and Paloma Sodhi and Ricky TQ Chen and Joseph Ortiz and Daniel DeTone and Austin Wang and Stuart Anderson and Jing Dong and Brandon Amos and Mustafa Mukadam},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```


## License

torchlie is MIT licensed. See the [LICENSE](https://github.com/facebookresearch/theseus/blob/main/torchlie/LICENSE) for details.


## Additional Information

- Join the community on [Github Discussions](https://github.com/facebookresearch/theseus/discussions) for questions and sugesstions.
- Use [Github Issues](https://github.com/facebookresearch/theseus/issues/new/choose) for bugs and features.
- See [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md) if interested in helping out.
