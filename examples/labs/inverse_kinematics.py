#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This example illustrates the four backward modes
# (unroll, implicit, truncated, and dlm)
# on a problem fitting a quadratic to data.

import os

import torch

from lie.functional import SE3
from theseus.labs.embodied.robot.forward_kinematics import Robot, get_forward_kinematics

dtype = torch.float64
URDF_REL_PATH = (
    "../../tests/theseus_tests/embodied/kinematics/data/panda_no_gripper.urdf"
)
urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)

robot = Robot.from_urdf_file(urdf_path, dtype)
selected_links = ["panda_virtual_ee_link"]
fk, jfk_b, jfk_s = get_forward_kinematics(robot, selected_links)

print("---------------------------------------------------")
print("Body Jacobian")
print("---------------------------------------------------")
targeted_theta = torch.rand(10, robot.dof, dtype=dtype)
targeted_poses_ee: torch.Tensor = fk(targeted_theta)[0]
theta_opt = torch.zeros_like(targeted_theta)
for iter in range(50):
    jac_b, poses = jfk_b(theta_opt)
    error = SE3.log(SE3.compose(SE3.inv(poses[-1]), targeted_poses_ee)).view(-1, 6, 1)
    print(error.norm())
    if error.norm() < 1e-4:
        break
    theta_opt = theta_opt + 0.5 * (jac_b[-1].pinverse() @ error).view(-1, robot.dof)

print("---------------------------------------------------")
print("Spatial")
print("---------------------------------------------------")
targeted_theta = torch.rand(10, robot.dof, dtype=dtype)
targeted_poses_ee = fk(targeted_theta)[0]
theta_opt = torch.zeros_like(targeted_theta)
for iter in range(50):
    jac_s, poses = jfk_s(theta_opt)
    error = SE3.log(SE3.compose(targeted_poses_ee, SE3.inv(poses[-1]))).view(-1, 6, 1)
    print(error.norm())
    if error.norm() < 1e-4:
        break
    theta_opt = theta_opt + 0.25 * (jac_s[-1].pinverse() @ error).view(-1, robot.dof)
