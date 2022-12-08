# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import List, Optional, Tuple, Type

import torch

from .cost_function import CostFunction
from .cost_weight import CostWeight
from .robust_loss import RobustLoss
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
# `RobustCostFunction.weighted_jacobians_error()`.
#
# Due to the nature of `RobustCostFunction`, its behavior is different to typical cost
# functions. Below we use the notation e, J, w, to refer to the error, jacobian, and
# weight of the base cost function, and let rho be the robust loss used. The important
# points are the following:
#
#   - For h := robust_cost_fn.weighted_error(), we have ||h||2 == rho(||w * e||2)
#   - For r_J, r_e defined as any jacobian/error returned by
#     robust_cost_fn.weighted_jacobians_error(), we have that
#     r_Jv^T * r_e == rho' * J^T * e, which is the gradient of rho(||w * e||2).
#   - Note that h != r_e. In general, weighted_jacobians_and_error()
#       is used by our optimizers, since it allows RobustCostFunction to be coupled
#       with any Jacobian-based NLS solver w/o modifications. However, if you are
#       interested in the robust cost value itself, you should use the error returned
#       by `weighted_error()` and **NOT** the one returned by
#       `weighted_jacobians_error()`.
#
# Finally, since we apply the weight before the robust loss, we adopt the convention
# that `robust_cost_fn.jacobians() == robust_cost_fn.weighted_jacobians_error()`, and
# `robust_cost_fn.error() == robust_cost_fn.weighed_error()`.
class RobustCostFunction(CostFunction):
    _EPS = 1e-20

    def __init__(
        self,
        cost_function: CostFunction,
        loss_cls: Type[RobustLoss],
        log_loss_radius: Variable,
        name: Optional[str] = None,
    ):
        self.cost_function = cost_function
        super().__init__(cost_function.weight, name=name)

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
        warnings.warn(
            "Computing the robust cost error requires weighting first, so "
            "error() is equivalent to weighted_error()."
        )
        return self.weighted_error()

    def weighted_error(self) -> torch.Tensor:
        weighted_error = self.cost_function.weighted_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        error_loss = self.loss.evaluate(squared_norm, self.log_loss_radius.tensor)

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
        warnings.warn(
            "Computing the robust cost error requires weighting first, so "
            "jacobians() is equivalent to weighted_jacobians_error()."
        )
        return self.weighted_jacobians_error()

    def weighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        (
            weighted_jacobians,
            weighted_error,
        ) = self.cost_function.weighted_jacobians_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        rescale = (
            self.loss.linearize(squared_norm, self.log_loss_radius.tensor)
            + RobustCostFunction._EPS
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

    @property
    def weight(self) -> CostWeight:
        return self.cost_function.weight

    @weight.setter
    def weight(self, weight: CostWeight):
        self.cost_function.weight = weight

    @property
    def _supports_masking(self) -> bool:
        return self.__supports_masking__

    @_supports_masking.setter
    def _supports_masking(self, val: bool):
        self.cost_function._supports_masking = val
        self.__supports_masking__ = val
