# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

import theseus as th

from .robust_loss import RobustLoss


class RobustCostFunction(th.CostFunction):
    def __init__(
        self,
        loss_function: RobustLoss,
        log_loss_radius: th.Vector,
        cost_function: th.CostFunction,
    ):
        if isinstance(cost_function, RobustCostFunction):
            raise ValueError(
                "{} is alreay a robust cost function.".format(cost_function.name)
            )

        self.loss_function = loss_function
        self.cost_function = cost_function
        self.log_loss_radius = log_loss_radius
        self._optim_vars_attr_names = self.cost_function._optim_vars_attr_names
        self._aux_vars_attr_names = self.cost_function._aux_vars_attr_names
        self.register_aux_var("log_loss_radius")

    def error(self) -> torch.Tensor:
        return self.cost_function.error()

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_function.jacobians()

    def weighted_error(self) -> torch.Tensor:
        return self.cost_function.weighted_error()

    def weighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_function.weighted_jacobians_error()

    def function_value(self) -> torch.Tensor:
        weighted_error = self.weighted_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        return self.loss_function.evaluate(squared_norm, loss_radius)

    def reweighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        weighted_jacobians, weighted_error = self.weighted_jacobians_error()
        squared_norm = torch.sum(weighted_error**2, dim=1, keepdim=True)
        loss_radius = torch.exp(self.log_loss_radius.data)
        rescale = self.loss_function.linearize(squared_norm, loss_radius).sqrt()

        return [
            (rescale * jacobian.view(jacobian.shape[0], -1)).view(jacobian.shape[0])
            for jacobian in weighted_jacobians
        ], rescale * weighted_error
