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
fk, jfk = get_forward_kinematics(robot, selected_links)

theta = torch.rand(10, robot.dof, dtype=dtype)
targeted_poses_ee: torch.Tensor = fk(theta)[0]

theta = torch.zeros_like(theta)

for iter in range(50):
    poses_ee: torch.Tensor = fk(theta)[0]
    error = SE3.log(SE3.compose(SE3.inv(poses_ee), targeted_poses_ee)).view(-1, 6, 1)
    print(error.norm())
    jac_w: torch.Tensor = jfk(theta)[0][-1]
    theta = theta + 0.2 * (jac_w.pinverse() @ error).view(-1, robot.dof)
