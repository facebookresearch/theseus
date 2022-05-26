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
Some other relevant files to look at:

* Pose Graph Optimization:
    - `examples/pose_graph_{cube/synthetic}.py`: Puts together optimization layer and implements outer loop. 

* Tactile State Estimation:
    - `utils/examples/tactile_pose_estimation/trainer.py`: Main outer learning loop.
    - `utils/examples/tactile_pose_estimation/pose_estimator.py`: Puts together the optimization layer.

* Bundle Adjustment:
    - `examples/bundle_adjustment.py`: Puts together optimization layer and implements outer loop. 

* Motion Planning:
    - `utils/examples/motion_planning/motion_planner.py`: Puts together optimization layer.
    - `examples/motion_planning_2d.py`: Implements outer loop.