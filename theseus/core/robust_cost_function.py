# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from ..geometry import Vector
from .cost_function import CostFunction
from .robust_loss import RobustLoss


class RobustCostFunction(CostFunction):
    def __init__(
        self,
        robust_loss: RobustLoss,
        log_loss_radius: Vector,
        cost_function: CostFunction,
        name: Optional[str] = None,
    ):
        if isinstance(cost_function, RobustCostFunction):
            raise ValueError(
                "{} is alreay a robust cost function.".format(cost_function.name)
            )

        super().__init__(cost_weight=cost_function.weight, name=name)
        self.robust_loss = robust_loss
        self.cost_function = cost_function
        self.log_loss_radius = log_loss_radius
        self.register_optim_vars(self.cost_function._optim_vars_attr_names)
        for attr in self._optim_vars_attr_names:
            setattr(self, attr, getattr(self.cost_function, attr))
        self.register_aux_vars(self.cost_function._aux_vars_attr_names)
        for attr in self._aux_vars_attr_names:
            setattr(self, attr, getattr(self.cost_function, attr))
        self.register_aux_var("log_loss_radius")

    def error(self) -> torch.Tensor:
        return self.cost_function.error()

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_function.jacobians()

    def weighted_error(self) -> torch.Tensor:
        return self.cost_function.weighted_error()

    def weighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_function.weighted_jacobians_error()

    def dim(self) -> int:
        return self.cost_function.dim()

    def function_value(self) -> torch.Tensor:
        weighted_error = self.weighted_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        return self.robust_loss.evaluate(squared_norm, loss_radius)

    def reweighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        weighted_jacobians, weighted_error = self.weighted_jacobians_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        rescale = self.robust_loss.linearize(squared_norm, loss_radius).sqrt()

        return [
            (rescale * jacobian.view(jacobian.shape[0], -1)).view(jacobian.shape)
            for jacobian in weighted_jacobians
        ], rescale * weighted_error

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.cost_function.to(*args, **kwargs)

    def _copy_impl(self, new_name: Optional[str] = None) -> "RobustCostFunction":
        return RobustCostFunction(
            robust_loss=self.robust_loss,
            log_loss_radius=self.log_loss_radius.copy(),
            cost_function=self.cost_function.copy(),
            name=new_name,
        )
