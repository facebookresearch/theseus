# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
__version__ = "0.1.4"

from .constants import DeviceType as DeviceType

from .core import (  # usort: skip
    AutoDiffCostFunction,
    AutogradMode,
    CostFunction,
    CostWeight,
    DiagonalCostWeight,
    HuberLoss,
    Objective,
    RobustCostFunction,
    RobustLoss,
    ScaleCostWeight,
    Variable,
    Vectorize,
    WelschLoss,
    as_variable,
    masked_jacobians,
    masked_variables,
)
from .geometry import (  # usort: skip
    LieGroup,
    LieGroupTensor,
    Manifold,
    Point2,
    Point3,
    SE2,
    SE3,
    SO2,
    SO3,
    Vector,
    adjoint,
    between,
    compose,
    enable_lie_group_check,
    enable_lie_tangent,
    exp_map,
    inverse,
    local,
    log_map,
    no_lie_group_check,
    no_lie_tangent,
    rand_point2,
    rand_point3,
    rand_se2,
    rand_se3,
    rand_so2,
    rand_so3,
    rand_vector,
    randn_point2,
    randn_point3,
    randn_se2,
    randn_se3,
    randn_so2,
    randn_so3,
    randn_vector,
    retract,
    set_lie_group_check_enabled,
    set_lie_tangent_enabled,
)
from .optimizer import (  # usort: skip
    DenseLinearization,
    Linearization,
    ManifoldGaussian,
    OptimizerInfo,
    SparseLinearization,
    VariableOrdering,
    local_gaussian,
    retract_gaussian,
)
from .optimizer.linear import (  # usort: skip
    BaspachoSparseSolver,
    CholeskyDenseSolver,
    CholmodSparseSolver,
    DenseSolver,
    LinearOptimizer,
    LinearSolver,
    LUCudaSparseSolver,
    LUDenseSolver,
)
from .optimizer.nonlinear import (  # usort: skip
    BackwardMode,
    DCEM,
    Dogleg,
    GaussNewton,
    LevenbergMarquardt,
    NonlinearLeastSquares,
    NonlinearOptimizerInfo,
    NonlinearOptimizerParams,
    NonlinearOptimizerStatus,
)
from .theseus_layer import TheseusLayer  # usort: skip

import theseus.embodied as eb  # usort: skip

# Aliases for some standard cost functions
Difference = eb.Local
Between = eb.Between
Local = eb.Local
