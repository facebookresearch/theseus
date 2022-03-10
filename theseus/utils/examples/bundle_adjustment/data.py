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
            torch.rand(proj_points.shape, dtype=proj_points.dtype) * 2 - 1
        )
    else:  # Normal(0, noiseSize)
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
    def __init__(
        self,
        num_points: int = 60,
        focal_length: float = 1000.0,
        dtype: torch.dtype = torch.float64,
    ):
        self.focal_length = torch.tensor([focal_length], dtype=dtype)

        # pts = [+/-10, +/-10, +/-1]
        self.world_points = torch.cat(
            [
                torch.rand(2, num_points, dtype=dtype) * 20 - 10,
                torch.rand(1, num_points, dtype=dtype) * 2 - 1,
            ]
        ).T

        # gt_cam_pos = [+/-3, +/-3, 5 +/-1]
        gt_cam_pos = th.Point3(
            torch.tensor(
                [
                    [
                        torch.rand((), dtype=dtype) * 3,
                        torch.rand((), dtype=dtype) * 3,
                        5 + torch.rand((), dtype=dtype),
                    ]
                ]
            )
        )
        self.gt_cam_rotation = th.SO3(
            random_small_quaternion(max_degrees=20, dtype=dtype)
        )
        self.gt_cam_translation = cast(
            th.Point3, -self.gt_cam_rotation.rotate(gt_cam_pos)
        )

        camera_points = (
            self.gt_cam_rotation.rotate(self.world_points) + self.gt_cam_translation
        )
        proj_points = (
            camera_points[:, :2] / camera_points[:, 2:3] * self.focal_length.data
        )

        # Here we add some noise around the ground truth values
        self.image_feature_points = add_noise_and_outliers(proj_points)
        small_rotation = th.SO3(random_small_quaternion(max_degrees=0.3))
        small_translation = torch.rand(3, dtype=dtype) * 0.1
        self.obs_cam_rotation = cast(
            th.SO3, small_rotation.compose(self.gt_cam_rotation)
        )
        self.obs_cam_translation = cast(
            th.Point3,
            small_rotation.rotate(self.gt_cam_translation) + small_translation,
        )


class LocalizationDataset:
    def __init__(self, num_samples: int, num_points: int, batch_size: int):
        self.samples = [
            LocalizationSample(num_points=num_points) for _ in range(num_samples)
        ]
        self.batch_size = batch_size
        self._cur_batch = 0

    def __iter__(self):
        self._cur_batch = 0
        return self

    def __next__(self):
        if self._cur_batch * self.batch_size >= len(self.samples):
            raise StopIteration

        batch_ls = self.samples[
            self._cur_batch * self.batch_size : (self._cur_batch + 1) * self.batch_size
        ]
        batch_data = {
            "camera_rotation": torch.cat([ls.obs_cam_rotation.data for ls in batch_ls]),
            "camera_translation": torch.cat(
                [ls.obs_cam_translation.data for ls in batch_ls]
            ),
            "focal_length": torch.cat(
                [ls.focal_length.data.unsqueeze(1) for ls in batch_ls]
            ),
        }

        # batch of 3d points and 2d feature points
        for i in range(len(batch_ls[0].world_points)):
            batch_data[f"world_point_{i}"] = torch.cat(
                [ls.world_points[i : i + 1].data for ls in batch_ls]
            )
            batch_data[f"image_feature_point_{i}"] = torch.cat(
                [ls.image_feature_points[i : i + 1].data for ls in batch_ls]
            )

        gt_cam_rotation = th.SO3(
            data=torch.cat([ls.gt_cam_rotation.data for ls in batch_ls])
        )
        gt_cam_translation = th.Point3(
            data=torch.cat([ls.gt_cam_translation.data for ls in batch_ls])
        )

        self._cur_batch += 1
        return batch_data, gt_cam_rotation, gt_cam_translation

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
