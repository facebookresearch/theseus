# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from theseus.core import CostFunction, CostWeight
from theseus.geometry import LieGroup, between


class MovingFrameBetween(CostFunction):
    def __init__(
        self,
        frame1: LieGroup,
        frame2: LieGroup,
        pose1: LieGroup,
        pose2: LieGroup,
        measurement: LieGroup,
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        seen_classes = set(
            [x.__class__.__name__ for x in [frame1, frame2, pose1, pose2, measurement]]
        )
        if len(seen_classes) > 1:
            raise ValueError("Inconsistent types between input variables.")

        super().__init__(cost_weight, name=name)
        self.frame1 = frame1
        self.frame2 = frame2
        self.pose1 = pose1
        self.pose2 = pose2
        self.register_optim_vars(["frame1", "frame2", "pose1", "pose2"])
        self.measurement = measurement
        self.register_aux_vars(["measurement"])

    def error(self) -> torch.Tensor:
        pose1__frame = between(self.frame1, self.pose1)
        pose2__frame = between(self.frame2, self.pose2)
        var_diff = between(pose1__frame, pose2__frame)
        return self.measurement.local(var_diff)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        jacobians_b1: List[torch.Tensor] = []
        pose1__frame = between(self.frame1, self.pose1, jacobians=jacobians_b1)
        jacobians_b2: List[torch.Tensor] = []
        pose2__frame = between(self.frame2, self.pose2, jacobians=jacobians_b2)
        jacobians_b_out: List[torch.Tensor] = []
        var_diff = between(pose1__frame, pose2__frame, jacobians=jacobians_b_out)
        error = self.measurement.local(var_diff)

        JB1_f1, JB1_p1 = jacobians_b1
        JB2_f2, JB2_p2 = jacobians_b2
        J_Bout_B1, J_Bout_B2 = jacobians_b_out
        J_out_f1 = torch.matmul(J_Bout_B1, JB1_f1)
        J_out_p1 = torch.matmul(J_Bout_B1, JB1_p1)
        J_out_f2 = torch.matmul(J_Bout_B2, JB2_f2)
        J_out_p2 = torch.matmul(J_Bout_B2, JB2_p2)

        return [J_out_f1, J_out_f2, J_out_p1, J_out_p2], error

    def dim(self) -> int:
        return self.frame1.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "MovingFrameBetween":
        return MovingFrameBetween(
            self.frame1.copy(),
            self.frame2.copy(),
            self.pose1.copy(),
            self.pose2.copy(),
            self.measurement.copy(),
            self.weight.copy(),
            name=new_name,
        )
