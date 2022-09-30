# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import torch

from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.embodied.kinematics import IdentityModel, KinematicsModel
from theseus.geometry import SE2, Point2

from .signed_distance_field import SignedDistanceField2D


class Collision2D(CostFunction):
    def __init__(
        self,
        pose: Union[Point2, SE2],
        sdf_origin: Union[Point2, torch.Tensor],
        sdf_data: Union[torch.Tensor, Variable],
        sdf_cell_size: Union[float, torch.Tensor, Variable],
        cost_eps: Union[float, Variable, torch.Tensor],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        if not isinstance(pose, Point2) and not isinstance(pose, SE2):
            raise ValueError("Collision2D only accepts Point2 or SE2 poses.")
        super().__init__(cost_weight, name=name)
        self.pose = pose
        self.sdf_origin = SignedDistanceField2D.convert_origin(sdf_origin)
        self.sdf_data = SignedDistanceField2D.convert_sdf_data(sdf_data)
        self.sdf_cell_size = SignedDistanceField2D.convert_cell_size(sdf_cell_size)
        self.cost_eps = as_variable(cost_eps)
        self.cost_eps.tensor = self.cost_eps.tensor.view(-1, 1)
        self.register_optim_vars(["pose"])
        self.register_aux_vars(["sdf_origin", "sdf_data", "sdf_cell_size", "cost_eps"])
        self.robot: KinematicsModel = IdentityModel()
        self.sdf = SignedDistanceField2D(
            self.sdf_origin, self.sdf_cell_size, self.sdf_data
        )

    def _compute_distances_and_jacobians(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        jac_xy: Optional[torch.Tensor] = None
        if isinstance(self.pose, SE2):
            aux: List[torch.Tensor] = []
            xy_pos = self.pose.xy(jacobians=aux)
            jac_xy = aux[0]
        else:
            xy_pos = cast(Point2, self.pose)

        robot_state = self.robot.forward_kinematics(xy_pos)["state"]
        dist, jac = self.sdf.signed_distance(robot_state.tensor.view(-1, 2, 1))
        if jac_xy is not None:
            jac = jac.matmul(jac_xy)
        return dist, jac

    def _error_from_distances(self, distances: torch.Tensor):
        return (self.cost_eps.tensor - distances).clamp(min=0)

    def error(self) -> torch.Tensor:
        distances, _ = self._compute_distances_and_jacobians()
        return self._error_from_distances(distances)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        distances, jacobian = self._compute_distances_and_jacobians()
        error = self._error_from_distances(distances)
        faraway_idx = distances > self.cost_eps.tensor
        jacobian[faraway_idx] = 0.0
        return [-jacobian], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "Collision2D":
        return Collision2D(
            self.pose.copy(),
            self.sdf_origin.copy(),
            self.sdf_data.copy(),
            self.sdf_cell_size.copy(),
            self.cost_eps.copy(),
            self.weight.copy(),
            name=new_name,
        )

    def dim(self) -> int:
        return 1

    # This is needed so that the SDF container also updates with the new aux var
    def set_aux_var_at(self, index: int, variable: Variable):
        super().set_aux_var_at(index, variable)
        self.sdf.update_data(self.sdf_origin, self.sdf_data, self.sdf_cell_size)
