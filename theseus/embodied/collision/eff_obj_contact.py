# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import torch

from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.embodied.kinematics import IdentityModel
from theseus.geometry import SE2, Point2

from .signed_distance_field import SignedDistanceField2D


class EffectorObjectContactPlanar(CostFunction):
    def __init__(
        self,
        obj: SE2,
        eff: SE2,
        sdf_origin: Union[Point2, torch.Tensor],
        sdf_data: Union[torch.Tensor, Variable],
        sdf_cell_size: Union[float, torch.Tensor, Variable],
        eff_radius: Union[float, Variable, torch.Tensor],
        cost_weight: CostWeight,
        name: Optional[str] = None,
        use_huber_loss: bool = False,
    ):
        super().__init__(cost_weight, name=name)
        self.obj = obj
        self.eff = eff
        self.sdf_origin = SignedDistanceField2D.convert_origin(sdf_origin)
        self.sdf_data = SignedDistanceField2D.convert_sdf_data(sdf_data)
        self.sdf_cell_size = SignedDistanceField2D.convert_cell_size(sdf_cell_size)
        self.eff_radius = as_variable(eff_radius)
        if self.eff_radius.tensor.squeeze().ndim > 1:
            raise ValueError("eff_radius must be a 0-D or 1-D tensor.")
        self.eff_radius.tensor = self.eff_radius.tensor.view(-1, 1)
        self.register_optim_vars(["obj", "eff"])
        self.register_aux_vars(
            ["sdf_origin", "sdf_data", "sdf_cell_size", "eff_radius"]
        )
        self.robot = IdentityModel()
        self.sdf = SignedDistanceField2D(
            self.sdf_origin, self.sdf_cell_size, self.sdf_data
        )
        self._use_huber = use_huber_loss

        if use_huber_loss:
            raise NotImplementedError(
                "Jacobians for huber loss are not yet implemented."
            )

    def _compute_distances_and_jacobians(
        self,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        J_transf: List[torch.Tensor] = []
        J_xy: List[torch.Tensor] = []
        eff__obj = self.obj.transform_to(
            self.eff.xy(jacobians=J_xy).tensor, jacobians=J_transf
        )
        J_transf_obj = J_transf[0]
        J_transf_eff = J_transf[1].matmul(J_xy[0])
        robot_state = cast(Point2, self.robot.forward_kinematics(eff__obj)["state"])
        dist, J_dist = self.sdf.signed_distance(robot_state.tensor.view(-1, 2, 1))
        J_out = (J_dist.matmul(J_transf_obj), J_dist.matmul(J_transf_eff))
        return dist, J_out

    def _error_from_distances(self, distances: torch.Tensor):
        if self._use_huber:
            eff_rad = self.eff_radius.tensor
            err = distances.clone()
            # linear (two-sided, otherwise this would be 0)
            gt_r_idx = distances >= 2 * eff_rad
            err[gt_r_idx] = distances[gt_r_idx] - 0.5 * eff_rad
            # quadratic
            lt_r_pos_idx = (distances < 2 * eff_rad).logical_or(distances > 0)
            err[lt_r_pos_idx] = (
                0.5 * (distances[lt_r_pos_idx] - eff_rad).square() / eff_rad
            )
            # linear
            neg_idx = distances < 0
            err[neg_idx] = -distances[neg_idx] + 0.5 * eff_rad
            return err
        else:
            return (distances - self.eff_radius.tensor).abs()

    def error(self) -> torch.Tensor:
        distances, _ = self._compute_distances_and_jacobians()
        return self._error_from_distances(distances)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        distances, jacobians = self._compute_distances_and_jacobians()
        error = self._error_from_distances(distances)
        if self._use_huber:
            raise NotImplementedError(
                "Jacobians for huber loss are not yet implemented."
            )
        else:
            lt_idx = distances < self.eff_radius.tensor
            jacobians[0][lt_idx] *= -1.0
            jacobians[1][lt_idx] *= -1.0
        return [jacobians[0], jacobians[1]], error

    def _copy_impl(
        self, new_name: Optional[str] = None
    ) -> "EffectorObjectContactPlanar":
        return EffectorObjectContactPlanar(
            self.obj.copy(),
            self.eff.copy(),
            self.sdf_origin.copy(),
            self.sdf_data.copy(),
            self.sdf_cell_size.copy(),
            self.eff_radius.copy(),
            self.weight.copy(),
            name=new_name,
        )

    def dim(self) -> int:
        return 1

    # This is needed so that the SDF container also updates with the new aux var
    def set_aux_var_at(self, index: int, variable: Variable):
        super().set_aux_var_at(index, variable)
        self.sdf.update_data(self.sdf_origin, self.sdf_data, self.sdf_cell_size)
