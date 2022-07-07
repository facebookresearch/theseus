# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

import theseus as th


class Reprojection(th.CostFunction):
    def __init__(
        self,
        camera_pose: th.SE3,
        world_point: th.Point3,
        image_feature_point: th.Point2,
        focal_length: th.Vector,
        calib_k1: th.Vector = None,
        calib_k2: th.Vector = None,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=camera_pose.dtype))
        super().__init__(
            cost_weight=weight,
            name=name,
        )
        self.camera_pose = camera_pose
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2
        batch_size = self.camera_pose.shape[0]
        if self.calib_k1 is None:
            self.calib_k1 = th.Vector(
                tensor=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name="calib_k1",
            )
        if self.calib_k2 is None:
            self.calib_k2 = th.Vector(
                tensor=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name="calib_k2",
            )
        self.world_point = world_point
        self.image_feature_point = image_feature_point

        self.register_optim_vars(["camera_pose", "world_point"])
        self.register_aux_vars(
            ["focal_length", "image_feature_point", "calib_k1", "calib_k2"]
        )

    def error(self) -> torch.Tensor:
        point_cam = self.camera_pose.transform_from(self.world_point)
        proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.tensor * (
            1.0 + proj_sqn * (self.calib_k1.tensor + proj_sqn * self.calib_k2.tensor)
        )
        point_projection = proj * proj_factor

        err = point_projection - self.image_feature_point.tensor
        return err

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cpose_wpt_jacs: List[torch.Tensor] = []
        point_cam = self.camera_pose.transform_from(self.world_point, cpose_wpt_jacs)
        J = torch.cat(cpose_wpt_jacs, dim=2)

        proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.tensor * (
            1.0 + proj_sqn * (self.calib_k1.tensor + proj_sqn * self.calib_k2.tensor)
        )
        d_proj_factor = self.focal_length.tensor * (
            self.calib_k1.tensor + 2.0 * proj_sqn * self.calib_k2.tensor
        )
        point_projection = proj * proj_factor

        # derivative of N/D is (N' - ND'/D) / D
        d_num = J[:, 0:2, :]
        num_dden_den = torch.bmm(
            point_cam[:, :2].unsqueeze(2),
            (J[:, 2, :] / point_cam[:, 2:3]).unsqueeze(1),
        )
        proj_jac = (num_dden_den - d_num) / point_cam[:, 2:].unsqueeze(2)
        proj_sqn_jac = 2.0 * proj.unsqueeze(2) * torch.bmm(proj.unsqueeze(1), proj_jac)
        point_projection_jac = proj_jac * proj_factor.unsqueeze(
            2
        ) + proj_sqn_jac * d_proj_factor.unsqueeze(2)

        err = point_projection - self.image_feature_point.tensor
        return [point_projection_jac[..., :6], point_projection_jac[..., 6:]], err

    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self, new_name: Optional[str] = None) -> "Reprojection":
        return Reprojection(
            self.camera_pose.copy(),
            self.world_point.copy(),
            self.image_feature_point.copy(),
            self.focal_length.copy(),
            calib_k1=self.calib_k1.copy(),
            calib_k2=self.calib_k2.copy(),
            weight=self.weight.copy(),
            name=new_name,
        )
