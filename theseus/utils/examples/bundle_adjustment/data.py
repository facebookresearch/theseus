from typing import List, Optional, cast

import numpy as np
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


class Camera:
    def __init__(
        self,
        pose: th.SE3,
        focal_length: th.Variable,
        calib_k1: th.Variable,
        calib_k2: th.Variable,
    ):
        self.pose = pose
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2

    def to_params(self) -> List[float]:
        batch_size = self.pose.data.shape[0]
        assert batch_size == 1
        R = th.SO3(data=self.pose.data[:, :, :3]).log_map().squeeze(0)
        t = self.pose.data[:, :, 3].squeeze(0)
        return [
            *R.numpy(),
            *t.numpy(),
            float(self.focal_length[0, 0]),
            float(self.calib_k1[0, 0]),
            float(self.calib_k2[0, 0]),
        ]

    @staticmethod
    def from_params(params: List[float], name: str = "Cam") -> "Camera":
        r = th.SO3.exp_map(torch.tensor(params[:3], dtype=torch.float64).unsqueeze(0))
        t = torch.tensor([params[3:6]], dtype=torch.float64).unsqueeze(2)
        pose = th.SE3(data=torch.cat([r.data, t], dim=2), name=name + "_pose")
        focal_length = th.Variable(
            data=torch.tensor([params[6:7]], dtype=torch.float64),
            name=name + "_focal_length",
        )
        calib_k1 = th.Variable(
            data=torch.tensor([params[7:8]], dtype=torch.float64),
            name=name + "_calib_k1",
        )
        calib_k2 = th.Variable(
            data=torch.tensor([params[8:9]], dtype=torch.float64),
            name=name + "_calib_k2",
        )
        return Camera(pose, focal_length, calib_k1, calib_k2)

    def project_point(self, point: th.Point3) -> torch.Tensor:
        point_cam = self.pose.transform_from(point)
        proj = -point_cam.data[:, :2] / point_cam.data[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.data * (
            1.0 + proj_sqn * (self.calib_k1.data + proj_sqn * self.calib_k2.data)
        )
        return proj * proj_factor

    def perturbed(self, rot_random: float = 0.1, pos_random: float = 0.2):
        batch_size = self.pose.shape[0]
        pert_rot = torch.cat(
            [
                random_small_quaternion(max_degrees=rot_random).unsqueeze(0)
                for _ in range(batch_size)
            ]
        )
        pert_tr = (
            torch.rand((batch_size, 3), dtype=torch.float64) * 2
            + torch.tensor([-1, -1, -1], dtype=torch.float64)
        ) * pos_random
        pert_data = torch.cat([pert_tr, pert_rot], dim=1)
        pert = th.SE3(pert_data)
        pert_pose = cast(th.SE3, pert.compose(self.pose))
        return Camera(
            pert_pose.copy(new_name=self.pose.name),
            self.focal_length.copy(new_name=self.focal_length.name),
            self.calib_k1.copy(new_name=self.calib_k1.name),
            self.calib_k2.copy(new_name=self.calib_k2.name),
        )

    @staticmethod
    def generate_synthetic(
        batch_size: int = 1,
        rot_random: float = 20.0,
        pos_random: float = 1.0,
        pos_base: torch.Tensor = torch.zeros(3, dtype=torch.float64),
        fl_random: float = 100.0,
        fl_base: float = 1000.0,
        k1_random: float = 0.1,
        k1_base: float = 0.0,
        k2_random: float = 0.05,
        k2_base: float = 0.0,
        name: str = "Cam",
    ):
        cam_rot = torch.cat(
            [
                random_small_quaternion(max_degrees=rot_random).unsqueeze(0)
                for _ in range(batch_size)
            ]
        )
        cam_tr = (
            torch.rand((batch_size, 3), dtype=torch.float64) * 2
            + torch.tensor([-1, -1, -1], dtype=torch.float64)
        ) * pos_random + pos_base
        cam_pose_data = torch.cat([cam_tr, cam_rot], dim=1)
        cam_pose = th.SE3(cam_pose_data, name=name + "_pose")

        focal_length = th.Vector(
            data=(torch.rand((batch_size, 1), dtype=torch.float64) * 2 - 1.0)
            * fl_random
            + fl_base,
            name=name + "_focal_length",
        )
        calib_k1 = th.Vector(
            data=(torch.rand((batch_size, 1), dtype=torch.float64) * 2 - 1.0)
            * k1_random
            + k1_base,
            name=name + "_calib_k1",
        )
        calib_k2 = th.Vector(
            data=(torch.rand((batch_size, 1), dtype=torch.float64) * 2 - 1.0)
            * k2_random
            + k2_base,
            name=name + "_calib_k2",
        )
        return Camera(cam_pose, focal_length, calib_k1, calib_k2)


class Observation:
    def __init__(
        self, camera_index: int, point_index: int, image_feature_point: th.Point2
    ):
        self.camera_index = camera_index
        self.point_index = point_index
        self.image_feature_point = image_feature_point


class BundleAdjustmentDataset:
    def __init__(
        self,
        cameras: List[Camera],
        points: List[th.Point3],
        observations: List[Observation],
        gt_cameras: Optional[List[Camera]] = None,
        gt_points: Optional[List[th.Point3]] = None,
    ):
        self.cameras = cameras
        self.points = points
        self.observations = observations
        self.gt_cameras = gt_cameras
        self.gt_points = gt_points

    @staticmethod
    def load_bal_dataset(path):
        observations = []
        cameras = []
        points = []
        with open(path, "rt") as out:
            num_cameras, num_points, num_observations = [
                int(x) for x in out.readline().rstrip().split()
            ]
            for i in range(num_observations):
                fields = out.readline().rstrip().split()
                feat = th.Point2(
                    data=torch.tensor(
                        [float(fields[2]), float(fields[3])], dtype=torch.float64
                    ).unsqueeze(0),
                    name=f"Feat{i}",
                )
                observations.append(
                    Observation(
                        camera_index=int(fields[0]),
                        point_index=int(fields[1]),
                        image_feature_point=feat,
                    )
                )

            for i in range(num_cameras):
                params = []
                for _ in range(9):
                    params.append(float(out.readline().rstrip()))
                cameras.append(Camera.from_params(params, name=f"Cam{i}"))

            for i in range(num_points):
                params = []
                for _ in range(3):
                    params.append(float(out.readline().rstrip()))
                points.append(
                    th.Point3(
                        data=torch.tensor(params, dtype=torch.float64).unsqueeze(0),
                        name=f"Pt{i}",
                    )
                )
        return cameras, points, observations

    @staticmethod
    def save_bal_dataset(path, cameras, points, observations):
        with open(path, "wt") as out:
            print(f"{len(cameras)} {len(points)} {len(observations)}", file=out)
            for obs in observations:
                f = obs.image_feature_point.data.squeeze(0).numpy()
                print(f"{obs.camera_index} {obs.point_index} {f[0]} {f[1]}", file=out)
            for cam in cameras:
                params = cam.to_params()
                for p in params:
                    print(f"{p}", file=out)
            for pt in points:
                params = pt.data.squeeze(0).numpy()
                for p in params:
                    print(f"{p}", file=out)

    @staticmethod
    def load_from_file(path, gt_path: Optional[str] = None):
        cameras, points, observations = BundleAdjustmentDataset.load_bal_dataset(path)
        if gt_path is not None:
            gt_points, gt_cameras, _ = BundleAdjustmentDataset.load_bal_dataset(gt_path)
        else:
            gt_points, gt_cameras = None, None
        return BundleAdjustmentDataset(
            cameras=cameras,
            points=points,
            observations=observations,
            gt_cameras=gt_cameras,
            gt_points=gt_points,
        )

    def save_to_file(self, path, gt_path: Optional[str] = None):
        BundleAdjustmentDataset.save_bal_dataset(
            path, self.cameras, self.points, self.observations
        )
        if gt_path is not None:
            BundleAdjustmentDataset.save_bal_dataset(
                gt_path, self.gt_cameras, self.gt_points, self.observations
            )

    def histogram(self):
        buckets = np.zeros(11)
        for obs in self.observations:
            proj_pt = self.cameras[obs.camera_index].project_point(
                self.points[obs.point_index]
            )
            error = float((obs.image_feature_point.data - proj_pt).norm())
            idx = min(int(error), len(buckets) - 1)
            buckets[idx] += 1
        max_buckets = max(buckets)
        for i in range(len(buckets)):
            bi = buckets[i]
            label = f"{i}-{i+1}" if i + 1 < len(buckets) else f"{i}+"
            barlen = round(bi * 80 / max_buckets)
            print(f"{label}: {'#' * barlen} {bi}")

    @staticmethod
    def generate_synthetic(
        num_cameras: int,
        num_points: int,
        average_track_length: int = 7,
        track_locality: float = 0.1,
        feat_random: float = 1.5,
        prob_feat_is_outlier: float = 0.02,
        outlier_feat_random: float = 70,
    ):

        # add cameras
        gt_cameras = [
            Camera.generate_synthetic(
                pos_base=torch.tensor(
                    [-i * 100.0 / (num_cameras - 1), 0, -100], dtype=torch.float64
                ),
                name=f"Cam{i}",
            )
            for i in range(num_cameras)
        ]
        cameras = [cam.perturbed() for cam in gt_cameras]

        # add points
        gt_points = [
            th.Point3(
                data=(torch.rand((1, 3), dtype=torch.float64) * 2 - 1) * 20
                + torch.tensor([i * 100.0 / (num_points), 0, 0], dtype=torch.float64),
                name=f"Pt{i}",
            )
            for i in range(num_points)
        ]
        points = [
            th.Point3(
                data=gt_points[i].data + (torch.rand((1, 3)) * 2 - 1) * 0.2,
                name=gt_points[i].name,
            )
            for i in range(num_points)
        ]

        # add observations
        pts_per_cam = average_track_length * num_points // num_cameras
        observations = []
        obs_num = 0
        for i in range(num_cameras):
            span_size = min(pts_per_cam + int(track_locality * num_points), num_points)
            span_start = (num_points - span_size) * i // num_cameras
            obs_pts = torch.randperm(span_size)[:pts_per_cam] + span_start
            # obs_pts = np.random.choice(np.arange(span_start, span_start + span_size), pts_per_cam)
            for j in obs_pts:
                # feat = gt_cameras[i].project_point(gt_points[j])
                feat = (
                    gt_cameras[i].project_point(gt_points[j])
                    + (torch.rand(1, 2) * 2 - 1) * feat_random
                )
                if torch.rand(()) < prob_feat_is_outlier:
                    feat = feat + (torch.rand(1, 2) * 2 - 1) * outlier_feat_random
                observations.append(
                    Observation(
                        camera_index=i,
                        point_index=j,
                        image_feature_point=th.Point2(data=feat, name=f"Feat{obs_num}"),
                    )
                )
                obs_num += 1

        return BundleAdjustmentDataset(
            cameras, points, observations, gt_cameras, gt_points
        )
