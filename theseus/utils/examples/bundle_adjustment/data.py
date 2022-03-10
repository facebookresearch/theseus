from typing import cast

import torch

import theseus as th

from .util import random_small_quaternion


def add_noise_and_outliers(
    proj_points: torch.Tensor,
    noise_size: int = 1,
    linear_noise: bool = True,
    outliers_proportion: float = 0.05,
    outlier_distance: float = 500.0,
) -> torch.Tensor:

    if linear_noise:
        feat_image_points = proj_points + noise_size * (
            torch.rand(proj_points.shape, dtype=torch.float64) * 2 - 1
        )
    else:  # normal, stdDev = noiseSize
        feat_image_points = proj_points + torch.normal(
            mean=torch.zeros(proj_points.shape), std=noise_size
        ).to(dtype=proj_points.dtype)

    # add real bad outliers
    outliers_mask = torch.rand(feat_image_points.shape[0]) < outliers_proportion
    num_outliers = feat_image_points[outliers_mask].shape[0]
    feat_image_points[outliers_mask] += outlier_distance * (
        torch.rand((num_outliers, proj_points.shape[1]), dtype=proj_points.dtype) * 2
        - 1
    )
    return feat_image_points


class LocalizationSample:
    def __init__(self, num_points: int = 60, focal_length: float = 1000.0):
        self.focal_length = torch.tensor([focal_length], dtype=torch.float64)

        # pts = [+/-10, +/-10, +/-1]
        self.world_points = torch.cat(
            [
                torch.rand(2, num_points, dtype=torch.float64) * 20 - 10,
                torch.rand(1, num_points, dtype=torch.float64) * 2 - 1,
            ]
        ).T

        # gt_cam_pos = [+/-3, +/-3, 5 +/-1]
        gt_cam_pos = th.Point3(
            torch.tensor(
                [
                    [
                        torch.rand((), dtype=torch.float64) * 3,
                        torch.rand((), dtype=torch.float64) * 3,
                        5 + torch.rand((), dtype=torch.float64),
                    ]
                ]
            )
        )
        self.gt_cam_rotation = th.SO3(random_small_quaternion(max_degrees=20))
        self.gt_cam_translation = cast(
            th.Point3, -self.gt_cam_rotation.rotate(gt_cam_pos)
        )

        camera_points = (
            self.gt_cam_rotation.rotate(self.world_points) + self.gt_cam_translation
        )
        proj_points = (
            camera_points[:, :2] / camera_points[:, 2:3] * self.focal_length.data
        )
        self.image_feature_points = add_noise_and_outliers(proj_points)

        small_rotation = th.SO3(random_small_quaternion(max_degrees=0.3))
        small_translation = torch.rand(3, dtype=torch.float64) * 0.1
        self.obs_cam_rotation = cast(
            th.SO3, small_rotation.compose(self.gt_cam_rotation)
        )
        self.obs_cam_translation = cast(
            th.Point3,
            small_rotation.rotate(self.gt_cam_translation) + small_translation,
        )
