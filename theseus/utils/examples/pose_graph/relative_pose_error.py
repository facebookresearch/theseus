# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch

import theseus as th

from .loss import Loss


class RelativePoseError(th.eb.Between):
    def __init__(
        self,
        v0: Union[th.SE2, th.SE3],
        v1: Union[th.SE2, th.SE3],
        cost_weight: th.CostWeight,
        measurement: Union[th.SE2, th.SE3],
        log_loss_radius: th.Vector,
        loss: Loss,
        name: Optional[str] = None,
    ):
        super().__init__(v0, v1, cost_weight, measurement, name)
        self.log_loss_radius = log_loss_radius
        self.loss = loss

        self.register_aux_vars(["log_loss_radius"])

    def weighted_error(self) -> torch.Tensor:
        return super().weighted_error()

    def weighted_jacobians_error(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return super().weighted_jacobians_error()
