# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch

from .cost_function import CostFunction
from .loss import Loss
from .variable import Variable


# This class implements a robust cost function that incorporates a loss for
# reducing the influence of outliers.
# See http://ceres-solver.org/nnls_modeling.html#lossfunction
#
# It works by wrapping a CostFunction, inheriting all of its optimization variables
# and auxiliary variables, without copying them. During linearization, the loss is
# applied to the error's squared norm, linearized, and used to rescale the underlying
# cost function's error and jacobians to solve for the full robust cost function
# (see Theory section at http://ceres-solver.org/nnls_modeling.html#theory and
# references therein); here we use alpha=0. The implementation of this part is done via
# `RobustCostFunction.weighted_jacobians_error()`
#
# Let e := `robust_cost_fn.cost_function.error()`. Currently, the convention is that:
#     -`robust_cost_fn.error()` returns e.
#     -`robust_cost_fn.weighted_error()` returns a vectorized version of loss(||e||^2).
#
# Also, `robust_cost_fn.jacobians()` is not implemented.
class RobustCostFunction(CostFunction):
    _EPS = 1e-20

    def __init__(
        self,
        cost_function: CostFunction,
        loss_cls: Type[Loss],
        log_loss_radius: Variable,
        name: Optional[str] = None,
    ):
        super().__init__(cost_function.weight, name=name)

        self.cost_function = cost_function
        # Register optimization variables of the underlying cost function
        for attr in cost_function._optim_vars_attr_names:
            setattr(self, attr, getattr(cost_function, attr))
            self.register_optim_var(attr)

        # Register auxiliary variables of the underlying cost function
        for attr in cost_function._aux_vars_attr_names:
            setattr(self, attr, getattr(cost_function, attr))
            self.register_aux_var(attr)

        self.log_loss_radius = log_loss_radius
        self.register_aux_var("log_loss_radius")
        self.loss = loss_cls()

    def error(self) -> torch.Tensor:
        return self.cost_function.error()

    def weighted_error(self) -> torch.Tensor:
        weighted_error = self.cost_function.weighted_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        error_loss = self.loss.evaluate(squared_norm, loss_radius)

        # The return value is a hacky way to make it so that
        # ||weighted_error||^2 = error_loss
        # By doing this we avoid having to change the objective's error computation
        # specifically for robust cost functions. The issue for this type of cost
        # function is that the theory requires us to maintain scaled errors/jacobians
        # of dim = robust_fn.cost_function.dim() to do the linearization properly,
        # but the actual error has dim = 1, being the result of loss(||error||^2).
        return (
            torch.ones_like(weighted_error)
            * (error_loss / self.dim() + RobustCostFunction._EPS).sqrt()
        )

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def weighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        (
            weighted_jacobians,
            weighted_error,
        ) = self.cost_function.weighted_jacobians_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        rescale = (
            self.loss.linearize(squared_norm, loss_radius) + RobustCostFunction._EPS
        ).sqrt()

        return [
            rescale.view(-1, 1, 1) * jacobian for jacobian in weighted_jacobians
        ], rescale * weighted_error

    def dim(self) -> int:
        return self.cost_function.dim()

    def _copy_impl(self, new_name: Optional[str] = None) -> "RobustCostFunction":
        return RobustCostFunction(
            self.cost_function.copy(),
            type(self.loss),
            self.log_loss_radius.copy(),
            name=new_name,
        )
