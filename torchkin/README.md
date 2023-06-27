# torchkin

<p align="center">
    <!-- License -->
    <a href="https://github.com/facebookresearch/theseus/blob/main/torchkin/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" height="20">
    </a>
    <!-- pypi -->
    <a href="https://pypi.org/project/torchkin/">
        <img src="https://img.shields.io/pypi/v/torchkin" alt="pypi"
        heigh="20">
    <!-- Downloads counter -->
    <a href="https://pypi.org/project/torchkin/">
        <img src="https://pepy.tech/badge/torchkin" alt="PyPi Downloads" height="20">
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
    <i>A library for differentiable kinematics</i>
</p>

-----

## Getting Started

### Prerequisites
- We *strongly* recommend you install torchkin in a venv or conda environment with Python 3.8-3.10.
- torchkin requires `torch` installation. To install for your particular CPU/CUDA configuration, follow the instructions in the PyTorch [website](https://pytorch.org/get-started/locally/).

### Installing

- **pypi**
    ```bash
    pip install torchkin
    ```

- #### **From source**
    The simplest way to install torchkin from source is by running the following
    ```bash
    git clone https://github.com/facebookresearch/theseus.git && cd theseus/torchkin
    pip install -e .
    ```
    If you are interested in contributing to torchkin, also install
    ```bash
    pip install -r ../requirements/dev.txt
    pre-commit install
    ```
    and follow the more detailed instructions in [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md).


## Example

An inverse kinematics example is available in [script](https://github.com/facebookresearch/theseus/blob/main/examples/inverse_kinematics.py).

```python
import torch

import torchkin as kin

# We can load a robot model from a URDF file
dtype = torch.float64
device = "cuda"
robot = kin.Robot.from_urdf_file(YOUR_URDF_FILE, dtype=dtype, device=device)

# Print robot name, number of links and degrees of freedom
print(f"{robot.name} has {len(robot.get_links())} links and {robot.dof} degrees of freedom.\n")

# Print joint id and name
for id, name in enumerate(robot.joint_map):
    # A joint is not fixed if and only if id < robot.dof
    print(f"joint {id}: {name} is {'not fixed' if id < robot.dof else 'fixed'}")
print("\n")

# Print link id and name
for link in robot.get_links():
    print(f"link {link.id}: {link.name}")

# We can get differentiable forward kinematics functions for specific links
# by using `get_forward_kinematics_fns`. This function creates three differentiable
# functions for evaluating forward kinematics, body jacobian and spatial jacobian of
# the selected links, in that order. The return types of these functions are as
# follows:
#
# - fk: return a tuple of link poses in the order of link names
# - jfk_b: returns a tuple where the first is a list of link body jacobians, and the
#          second is a tuple of link poses---both are in the order of link names
# - jfk_s: same as jfk_b except returning the spatial jacobians
link_names = [LINK1, LINK2, LINK3]
fk, jfk_b, jfk_s = kin.get_forward_kinematics_fns(
    robot=robot, link_names=link_names)

batch_size = 10
# The joint states are in the order of the joint ids
joint_states = torch.rand(batch_size, robot.dof, dtype=dtype, device=device)

# Get link poses
link_poses = fk(joint_states)

# Get body jacobians and link poses
jacs_b, link_poses = jfk_b(joint_states) 

# Get spatial jacobians and link poses
jacs_s, link_poses = jfk_s(joint_states) 
```

## Citing torchkin

If you use torchkin in your work, please cite the [paper](https://arxiv.org/abs/2207.09442) with the BibTeX below.

```bibtex
@article{pineda2022theseus,
  title   = {{Theseus: A Library for Differentiable Nonlinear Optimization}},
  author  = {Luis Pineda and Taosha Fan and Maurizio Monge and Shobha Venkataraman and Paloma Sodhi and Ricky TQ Chen and Joseph Ortiz and Daniel DeTone and Austin Wang and Stuart Anderson and Jing Dong and Brandon Amos and Mustafa Mukadam},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```

## License

torchkin is MIT licensed. See the [LICENSE](https://github.com/facebookresearch/theseus/blob/main/torchkin/LICENSE) for details.


## Additional Information

- Join the community on [Github Discussions](https://github.com/facebookresearch/theseus/discussions) for questions and sugesstions.
- Use [Github Issues](https://github.com/facebookresearch/theseus/issues/new/choose) for bugs and features.
- See [CONTRIBUTING](https://github.com/facebookresearch/theseus/blob/main/CONTRIBUTING.md) if interested in helping out.
