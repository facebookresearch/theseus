# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import os
import torch
from theseus.geometry.functional.constants import TEST_EPS
from theseus.embodied.robot.forward_kinematics import Robot
from theseus.embodied.robot.forward_kinematics import (
    get_forward_kinematics,
    ForwardKinematicsFactory,
)


URDF_REL_PATH = "../kinematics/data/panda_no_gripper.urdf"
urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)


@pytest.mark.parametrize("batch_size", [1, 20, 40])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp(batch_size: int, dtype: torch.dtype):
    robot = Robot.from_urdf_file(urdf_path, dtype)
    selected_links = ["panda_link2", "panda_link5", "panda_virtual_ee_link"]
    _, fkin_impl, _, _, _ = ForwardKinematicsFactory(robot, selected_links)
    fkin, _ = get_forward_kinematics(robot, selected_links)

    rng = torch.Generator()
    rng.manual_seed(0)
    angles = torch.rand(batch_size, robot.dof, generator=rng, dtype=dtype)

    jacs_impl = torch.autograd.functional.jacobian(fkin_impl, angles)
    jacs = torch.autograd.functional.jacobian(fkin, angles)

    for jac, jac_impl in zip(jacs, jacs_impl):
        assert torch.allclose(jac_impl, jac, atol=TEST_EPS)
