from typing import List, Optional, Tuple

import torch

import theseus as th

from .util import soft_loss_huber_like


class ReprojectionError(th.CostFunction):
    def __init__(
        self,
        camera_rotation: th.SO3,
        camera_translation: th.Point3,
        focal_length: th.Vector,
        loss_radius: th.Vector,
        world_point: th.Point3,
        image_feature_point: th.Point3,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(
                torch.tensor(1.0).to(dtype=camera_rotation.dtype)
            )
        super().__init__(
            cost_weight=weight,
            name=name,
        )
        self.camera_rotation = camera_rotation
        self.camera_translation = camera_translation
        self.loss_radius = loss_radius
        self.focal_length = focal_length
        self.world_point = world_point
        self.image_feature_point = image_feature_point

        self.register_optim_vars(["camera_rotation", "camera_translation"])
        self.register_aux_vars(
            ["loss_radius", "focal_length", "image_feature_point", "world_point"]
        )

    def error(self) -> torch.Tensor:
        point__cam = (
            self.camera_rotation.rotate(self.world_point) + self.camera_translation
        )
        point_projection = (
            point__cam[:, :2] / point__cam[:, 2:3] * self.focal_length.data
        )
        err = point_projection - self.image_feature_point.data

        err_norm = torch.norm(err, dim=1).unsqueeze(1)
        exp_loss = torch.exp(self.loss_radius.data)

        val, _ = soft_loss_huber_like(err_norm, exp_loss)
        return val

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        point__cam = (
            self.camera_rotation.rotate(self.world_point) + self.camera_translation
        )
        batch_size = self.camera_rotation.shape[0]
        X = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
        X[:, 0, 1] = self.world_point[:, 2]
        X[:, 0, 2] = -self.world_point[:, 1]
        X[:, 1, 0] = -self.world_point[:, 2]
        X[:, 1, 2] = self.world_point[:, 0]
        X[:, 2, 0] = self.world_point[:, 1]
        X[:, 2, 1] = -self.world_point[:, 0]
        J = torch.cat(
            (
                torch.bmm(self.camera_rotation.data, X),
                torch.eye(3, 3).unsqueeze(0).repeat(batch_size, 1, 1),
            ),
            dim=2,
        )

        point_projection = (
            point__cam[:, :2] / point__cam[:, 2:]
        ) * self.focal_length.data
        d_num = J[:, 0:2, :]
        num_dden_den = torch.bmm(
            point__cam[:, :2].unsqueeze(2),
            (J[:, 2, :] / point__cam[:, 2:3]).unsqueeze(1),
        )
        d_proj = (
            (d_num - num_dden_den)
            / point__cam[:, 2:].unsqueeze(2)
            * self.focal_length.data.unsqueeze(2)
        )
        err = point_projection - self.image_feature_point.data

        err_norm = torch.norm(err, dim=1).unsqueeze(1)
        err_dir = err / err_norm
        norm_jac = torch.bmm(err_dir.unsqueeze(1), d_proj)
        exp_loss = torch.exp(self.loss_radius.data)

        val, der = soft_loss_huber_like(err_norm, exp_loss)
        soft_jac = norm_jac * der.unsqueeze(1)

        return [soft_jac[:, :, :3], soft_jac[:, :, 3:]], val

    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self):
        return ReprojectionError(
            self.camera_rotation,
            self.camera_translation,
            self.loss_radius,
            self.focal_length,
            self.world_point,
            self.image_feature_point,
            weight=self.weight,
            name=self.name,
        )
