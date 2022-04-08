import abc
from typing import List, Optional, Tuple, Type

import torch

import theseus as th

from .loss import Loss


# This class implements a robust cost function that incorporates a loss for
# reducing the influence of outliers.
# See http://ceres-solver.org/nnls_modeling.html#lossfunction
#
# It works by wrapping a CostFunction, inheriting all of its optimization variables
# and auxiliary variables, without copying them. During linearization, the loss is
# applied to the error's squared norm, linearized, and used to rescale the underlying
# cost function's error and jacobians to solve for the full robust cost function
# (see Theory section in the Ceres link above). This is done via
# `RobustCostFunction.weighted_jacobians_error()`
#
# Currently, the convention is that `RobustCostFunction.error()` and
# `RobustCostFunction.jacobians()` just forward the underlying cost function's methods.
class RobustCostFunction(th.CostFunction, abc.ABC):
    def __init__(
        self,
        cost_function: th.CostFunction,
        cost_weight: th.CostWeight,
        loss_cls: Type[Loss],
        log_loss_radius: th.Vector,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)

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

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_function.jacobians()

    def weighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        (
            weighted_jacobians,
            weighted_error,
        ) = self.cost_function.weighted_jacobians_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        rescale = self.loss.linearize(squared_norm, loss_radius)

        return [
            (rescale * jacobian.view(jacobian.shape[0], -1)).view(jacobian.shape[0])
            for jacobian in weighted_jacobians
        ], rescale * weighted_error

    def dim(self) -> int:
        return self.cost_function.dim()

    def _copy_impl(self, new_name: Optional[str] = None) -> "RobustCostFunction":
        return RobustCostFunction(
            self.cost_function,
            self.weight.copy(),
            type(self.loss),
            self.log_loss_radius.copy(),
            name=new_name,
        )
