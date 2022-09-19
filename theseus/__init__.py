# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.1.0"

from .core import (
    CostFunction,
    CostWeight,
    DiagonalCostWeight,
    Objective,
    ScaleCostWeight,
    Variable,
    Vectorize,
    RobustLoss,
    AutoDiffCostFunction,
    AutogradMode,
    RobustCostFunction,
    HuberLoss,
    WelschLoss,
    as_variable,
)
from .geometry import (
    SE2,
    SE3,
    SO2,
    SO3,
    LieGroup,
    Manifold,
    Point2,
    Point3,
    Vector,
    adjoint,
    between,
    local,
    retract,
    compose,
    inverse,
    log_map,
    exp_map,
    LieGroupTensor,
    set_lie_tangent_enabled,
    enable_lie_tangent,
    no_lie_tangent,
    rand_vector,
    rand_point2,
    rand_point3,
    rand_so2,
    rand_so3,
    rand_se2,
    rand_se3,
    randn_vector,
    randn_point2,
    randn_point3,
    randn_so2,
    randn_so3,
    randn_se2,
    randn_se3,
)
from .optimizer import (
    DenseLinearization,
    OptimizerInfo,
    SparseLinearization,
    VariableOrdering,
    ManifoldGaussian,
    local_gaussian,
    retract_gaussian,
)
from .optimizer.linear import (
    CholeskyDenseSolver,
    CholmodSparseSolver,
    DenseSolver,
    LinearOptimizer,
    LUCudaSparseSolver,
    LUDenseSolver,
)
from .optimizer.nonlinear import (
    GaussNewton,
    LevenbergMarquardt,
    NonlinearLeastSquares,
    NonlinearOptimizerParams,
    NonlinearOptimizerStatus,
    BackwardMode,
)
from .theseus_layer import TheseusLayer

import theseus.embodied as eb

# Aliases for some standard cost functions
Difference = eb.Local
Between = eb.Between
Local = eb.Local
