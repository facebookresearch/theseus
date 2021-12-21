# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .core import (
    CostFunction,
    CostWeight,
    DiagonalCostWeight,
    AutoDiffCostFunction,
    Objective,
    ScaleCostWeight,
    Variable,
)
from .geometry import (
    SE2,
    SO2,
    LieGroup,
    Manifold,
    Point2,
    Point3,
    Vector,
    local,
    retract,
    compose,
    inverse,
    log_map,
    exp_map,
)
from .optimizer import DenseLinearization, SparseLinearization, VariableOrdering
from .optimizer.linear import (
    CholeskyDenseSolver,
    CholmodSparseSolver,
    DenseSolver,
    LinearOptimizer,
    LUDenseSolver,
)
from .optimizer.nonlinear import (
    GaussNewton,
    LevenbergMarquardt,
    NonlinearLeastSquares,
    NonlinearOptimizerParams,
    NonlinearOptimizerStatus,
)
from .theseus_layer import TheseusLayer
from .utils import random_sparse_binary_matrix

import theseus.embodied as eb

import torch

if torch.cuda.is_available():
    from .optimizer.linear import LUCudaSparseSolver
