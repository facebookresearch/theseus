# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, cast

import torch

from theseus.core import CostFunction, CostWeight, Variable
from theseus.embodied.kinematics import IdentityModel, RobotModel
from theseus.geometry import Point2

from .signed_distance_field import SignedDistanceField2D


class Collision2D(CostFunction):
    def __init__(
        self,
        pose: Point2,
        cost_weight: CostWeight,
        sdf_origin: Variable,
        sdf_data: Variable,
        sdf_cell_size: Variable,
        cost_eps: Variable,
        name: Optional[str] = None,
    ):
        if not isinstance(pose, Point2):
            raise ValueError("Collision2D only accepts 2D poses as inputs.")
        super().__init__(cost_weight, name=name)
        self.pose = pose
        self.sdf_origin = sdf_origin
        self.sdf_data = sdf_data
        self.sdf_cell_size = sdf_cell_size
        self.cost_eps = cost_eps
        self.register_optim_vars(["pose"])
        self.register_aux_vars(["sdf_origin", "sdf_data", "sdf_cell_size", "cost_eps"])
        self.robot: RobotModel = IdentityModel()
        self.sdf = SignedDistanceField2D(sdf_origin, sdf_cell_size, sdf_data)

    def _compute_distances_and_jacobians(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        robot_state = cast(Point2, self.robot.forward_kinematics(self.pose)["state"])
        return self.sdf.signed_distance(robot_state.data.view(-1, 2, 1))

    def _error_from_distances(self, distances: torch.Tensor):
        return (self.cost_eps.data - distances).clamp(min=0)

    def error(self) -> torch.Tensor:
        distances, _ = self._compute_distances_and_jacobians()
        return self._error_from_distances(distances)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        distances, jacobian = self._compute_distances_and_jacobians()
        error = self._error_from_distances(distances)
        faraway_idx = distances > self.cost_eps.data
        jacobian[faraway_idx] = 0.0
        return [-jacobian], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "Collision2D":
        return Collision2D(
            self.pose.copy(),
            self.weight.copy(),
            self.sdf_origin.copy(),
            self.sdf_data.copy(),
            self.sdf_cell_size.copy(),
            self.cost_eps.copy(),
            name=new_name,
        )

    def dim(self) -> int:
        return self.robot.dim()
