# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch

from theseus.core import CostFunction, CostWeight
from theseus.geometry import SE2, Point3, Vector


class Nonholonomic(CostFunction):
    def __init__(
        self,
        pose: Union[SE2, Point3, Vector],
        vel: Union[Point3, Vector],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if vel.dof() != 3 or pose.dof() != 3:
            raise ValueError(
                "Nonholonomic only accepts 3D velocity or poses (x, y, z dims). "
                "Poses can either be SE2 or Vector variables. Velocities only Vector."
            )
        self.pose = pose
        self.vel = vel
        self.register_optim_vars(["pose", "vel"])
        self.weight = cost_weight

    def dim(self):
        return 1

    def _compute_error(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vel = self.vel.tensor
        if isinstance(self.pose, SE2):
            error = vel[:, 1]
            J_error_pose = error.new_zeros(error.shape[0], 1, 3)
            J_error_vel = error.new_zeros(error.shape[0], 1, 3)
            J_error_vel[:, 0, 1] = 1
        else:
            cos = self.pose[:, 2].cos()
            sin = self.pose[:, 2].sin()
            error = vel[:, 1] * cos - vel[:, 0] * sin
            # Computing jacobians
            J_error_pose = self.vel.tensor.new_zeros(vel.shape[0], 1, 3)
            J_error_vel = self.vel.tensor.new_zeros(vel.shape[0], 1, 3)
            J_error_pose[:, 0, 2] = -(vel[:, 1] * sin + vel[:, 0] * cos)
            J_error_vel[:, 0, 0] = -sin
            J_error_vel[:, 0, 1] = cos
        return error.view(-1, 1), J_error_pose, J_error_vel

    def error(self) -> torch.Tensor:
        return self._compute_error()[0]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Pre-allocate jacobian tensors
        batch_size = self.vel.shape[0]
        Jvel = self.vel.tensor.new_zeros(batch_size, 1, 3)

        error, Jpose, Jvel = self._compute_error()
        return [Jpose, Jvel], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "Nonholonomic":
        return Nonholonomic(
            self.pose.copy(),
            self.vel.copy(),
            self.weight.copy(),
            name=new_name,
        )
