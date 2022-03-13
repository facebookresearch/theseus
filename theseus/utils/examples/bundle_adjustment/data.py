from typing import cast, List, Optional

import torch
import numpy as np

import theseus as th

# from .util import random_small_quaternion
from theseus.utils.examples.bundle_adjustment.util import random_small_quaternion


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


class Camera:
    def __init__(self,
                 pose: th.SE3,
                 focal_length: th.Variable,
                 calib_k1: th.Variable,
                 calib_k2: th.Variable):
        self.pose = pose
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2

    def project_point(self, point: th.Point3):
        point_cam = self.pose.transform_from(point)
        proj = point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.data * (
            1.0 + proj_sqn * (self.calib_k1.data + proj_sqn * self.calib_k2.data)
        )
        return proj * proj_factor

    def perturbed(self, rot_random:float=2, pos_random:float=0.5):
        batch_size = self.pose.shape[0]
        pert_rot = torch.cat(
            [
                random_small_quaternion(max_degrees=rot_random).unsqueeze(0)
                for _ in range(batch_size)
            ]
        )
        pert_tr = (torch.rand((batch_size, 3), dtype=torch.float64) * 2 
            + torch.tensor([-1, -1, -1], dtype=torch.float64)) * pos_random
        pert_data = torch.cat([pert_tr, pert_rot], dim=1)
        pert = th.SE3(pert_data)
        pert_pose = cast(th.SE3, pert.compose(self.pose))
        return Camera(pert_pose.copy(new_name=self.pose.name),
                      self.focal_length.copy(new_name=self.focal_length.name),
                      self.calib_k1.copy(new_name=self.calib_k1.name),
                      self.calib_k2.copy(new_name=self.calib_k2.name))

    @staticmethod
    def generate_synthetic(batch_size:int=1,
                           rot_random:float=20., 
                           pos_random:float=1.0,
                           pos_base:torch.Tensor=torch.zeros(3, dtype=torch.float64),
                           fl_random:float=100.,
                           fl_base:float=1000.,
                           k1_random:float=0.1,
                           k1_base:float=0.0,
                           k2_random:float=0.05,
                           k2_base:float=0.0,
                           name:str="Cam"):
        cam_rot = torch.cat(
            [
                random_small_quaternion(max_degrees=rot_random).unsqueeze(0)
                for _ in range(batch_size)
            ]
        )
        cam_tr = ((torch.rand((batch_size, 3), dtype=torch.float64) * 2 
            + torch.tensor([-1, -1, -1], dtype=torch.float64)) * pos_random
            + pos_base)
        cam_pose_data = torch.cat([cam_tr, cam_rot], dim=1)
        cam_pose = th.SE3(cam_pose_data, name=name+"_pose")

        focal_length = th.Vector(
            data=(torch.rand((batch_size,1)) * 2 - 1.0) * fl_random + fl_base,
            name=name+"_focal_length",
        )
        calib_k1 = th.Vector(
            data=(torch.rand((batch_size,1)) * 2 - 1.0) * k1_random + k1_base,
            name=name+"_calib_k1",
        )
        calib_k2 = th.Vector(
            data=(torch.rand((batch_size,1)) * 2 - 1.0) * k2_random + k2_base,
            name=name+"_calib_k2",
        )
        return Camera(cam_pose, focal_length, calib_k1, calib_k2)

class Observation:
    def __init__(self,
                 camera_index: int,
                 point_index: int,
                 image_feature_point: th.Point2):
        self.camera_index = camera_index
        self.point_index = point_index
        self.image_feature_point = image_feature_point


class BundleAdjustmentDataset:
    def __init__(self,
                 cameras: List[Camera],
                 points: List[th.Point3],
                 observations: List[Observation],
                 gt_cameras: Optional[List[Camera]] = None,
                 gt_points: Optional[List[th.Point3]] = None):
        self.cameras = cameras
        self.points = points
        self.observations = observations

    @staticmethod
    def generate_synthetic(num_cameras: int,
                           num_points: int,
                           average_track_length: int = 7,
                           track_locality: float = 0.1):
        
        # add cameras
        gt_cameras = [
            Camera.generate_synthetic(pos_base=torch.tensor([-i * 100.0 / (num_cameras-1), 0, 100],
                                      dtype=torch.float64),
                                      name=f"Cam{i}")
            for i in range(num_cameras)
        ]
        cameras = [
            cam.perturbed() for cam in gt_cameras
        ]

        # add points
        gt_points = [
            th.Point3(data=(torch.rand((1,3),dtype=torch.float64)*2 - 1)*20 + 
                            torch.tensor([i * 100.0 / (num_points),0,0],dtype=torch.float64),
                      name=f"Pt{i}")
            for i in range(num_points)
        ]
        points = [
            th.Point3(data=gt_points[i].data + (torch.rand((1,3))*2-1) * 0.4,
                      name=gt_points[i].name)
            for i in range(num_points)
        ]

        # add observations
        pts_per_cam = average_track_length * num_points // num_cameras
        observations = []
        for i in range(num_cameras):
            span_size = min(pts_per_cam + int(track_locality * num_points), num_points)
            span_start = (num_points - span_size) * i // num_cameras
            obs_pts = np.random.choice(np.arange(span_start, span_start + span_size), pts_per_cam)
            for j in obs_pts:
                feat = gt_cameras[i].project_point(gt_points[j]) + (torch.rand(1,2)*2-1)*0.8
                if np.random.randint(50) == 0:
                    feat = feat + (torch.rand(1,2)*2-1)*50
                observations.append(Observation(camera_index=i, 
                                                point_index=j,
                                                image_feature_point=feat))


BundleAdjustmentDataset.generate_synthetic(30, 1000)