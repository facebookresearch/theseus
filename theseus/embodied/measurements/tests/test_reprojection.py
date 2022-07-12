# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import theseus as th
from theseus.utils.examples.bundle_adjustment.util import random_small_quaternion


def test_residual():
    # unit test for Cost term
    torch.manual_seed(0)
    batch_size = 4
    cam_rot = torch.cat(
        [
            random_small_quaternion(max_degrees=20).unsqueeze(0)
            for _ in range(batch_size)
        ]
    )
    cam_tr = torch.rand((batch_size, 3), dtype=torch.float64) * 2 + torch.tensor(
        [-1, -1, -5.0], dtype=torch.float64
    )
    cam_pose_data = torch.cat([cam_tr, cam_rot], dim=1)
    cam_pose = th.SE3(cam_pose_data, name="cam_pose")

    focal_length = th.Vector(
        tensor=torch.tensor([1000], dtype=torch.float64)
        .repeat(batch_size)
        .unsqueeze(1),
        name="focal_length",
    )
    calib_k1 = th.Vector(
        tensor=torch.tensor([-0.1], dtype=torch.float64)
        .repeat(batch_size)
        .unsqueeze(1),
        name="calib_k1",
    )
    calib_k2 = th.Vector(
        tensor=torch.tensor([0.01], dtype=torch.float64)
        .repeat(batch_size)
        .unsqueeze(1),
        name="calib_k2",
    )
    world_point = th.Vector(
        tensor=torch.rand((batch_size, 3), dtype=torch.float64), name="worldPoint"
    )
    point_cam = cam_pose.transform_from(world_point).tensor
    proj = -point_cam[:, :2] / point_cam[:, 2:3]
    proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
    proj_factor = focal_length.tensor * (
        1.0 + proj_sqn * (calib_k1.tensor + proj_sqn * calib_k2.tensor)
    )
    point_projection = proj * proj_factor
    image_feature_point = th.Vector(
        tensor=point_projection + (torch.rand((batch_size, 2)) - 0.5) * 50,
        name="image_feature_point",
    )
    r = th.eb.Reprojection(
        camera_pose=cam_pose,
        world_point=world_point,
        focal_length=focal_length,
        calib_k1=calib_k1,
        calib_k2=calib_k2,
        image_feature_point=image_feature_point,
    )

    base_err = r.error()
    base_camera_pose = r.camera_pose.copy()
    base_world_point = r.world_point.copy()

    n_err = base_err.shape[1]
    pose_num_jac = torch.zeros((batch_size, n_err, 6), dtype=torch.float64)
    epsilon = 1e-8
    for i in range(6):
        v = torch.zeros((batch_size, 6), dtype=torch.float64)
        v[:, i] += epsilon
        r.camera_pose = base_camera_pose.retract(v)
        pert_err = r.error()
        pose_num_jac[:, :, i] = (pert_err - base_err) / epsilon
    r.camera_pose = base_camera_pose

    wpt_num_jac = torch.zeros((batch_size, n_err, 3), dtype=torch.float64)
    for i in range(3):
        v = torch.zeros((batch_size, 3), dtype=torch.float64)
        v[:, i] += epsilon
        r.world_point = base_world_point.retract(v)
        pert_err = r.error()
        wpt_num_jac[:, :, i] = (pert_err - base_err) / epsilon

    (pose_jac, wpt_jac), _ = r.jacobians()

    assert torch.norm(pose_num_jac - pose_jac) < 5e-5
    assert torch.norm(wpt_num_jac - wpt_jac) < 5e-5
