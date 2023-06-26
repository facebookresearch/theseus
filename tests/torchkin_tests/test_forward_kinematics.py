# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import os
import torch

from torchkin.forward_kinematics import (
    Robot,
    get_forward_kinematics_fns,
    ForwardKinematicsFactory,
)
from torchlie.functional import SE3
from torchlie.functional.constants import TEST_EPS

URDF_REL_PATH = "data/panda_no_gripper.urdf"
urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)


@pytest.mark.parametrize("batch_size", [1, 20, 40])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backward(batch_size: int, dtype: torch.dtype):
    robot = Robot.from_urdf_file(urdf_path, dtype)
    selected_links = ["panda_link2", "panda_link5", "panda_virtual_ee_link"]
    _, fk_impl, *_ = ForwardKinematicsFactory(robot, selected_links)
    fk, *_ = get_forward_kinematics_fns(robot, selected_links)

    rng = torch.Generator()
    rng.manual_seed(0)
    angles = torch.rand(batch_size, robot.dof, generator=rng, dtype=dtype)

    jacs_impl = torch.autograd.functional.jacobian(fk_impl, angles, vectorize=True)
    jacs = torch.autograd.functional.jacobian(fk, angles, vectorize=True)

    for jac, jac_impl in zip(jacs, jacs_impl):
        torch.testing.assert_close(
            actual=jac_impl, expected=jac, atol=TEST_EPS, rtol=1e-5
        )

    grads = []
    for func in [fk, fk_impl]:
        temp = angles.clone()
        temp.requires_grad = True
        loss = torch.tensor(0, dtype=dtype)
        for pose in func(temp):
            loss = loss + (pose**2).sum()
        loss.backward()
        grads.append(temp.grad)
    torch.testing.assert_close(actual=grads[0], expected=grads[1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 20, 40])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_jacobian(batch_size: int, dtype: torch.dtype):
    robot = Robot.from_urdf_file(urdf_path, dtype)
    selected_links = ["panda_link2", "panda_link5", "panda_virtual_ee_link"]
    fk, jfk_b, _ = get_forward_kinematics_fns(robot, selected_links)

    rng = torch.Generator()
    rng.manual_seed(0)
    angles = torch.rand(batch_size, robot.dof, generator=rng, dtype=dtype)

    jacs_actual, poses = jfk_b(angles)

    sels = range(batch_size)
    jacs_dense = torch.autograd.functional.jacobian(fk, angles, vectorize=True)
    jacs_expected = []
    for pose, jac_dense in zip(poses, jacs_dense):
        jac_dense = jac_dense[sels, :, :, sels].transpose(-1, 1).transpose(-1, -2)
        jac_expected = SE3.left_project(pose, jac_dense).transpose(-1, -2)
        jac_expected[:, 3:] *= 0.5
        jacs_expected.append(jac_expected)

    for jac_actual, jac_expected in zip(jacs_actual, jacs_expected):
        torch.testing.assert_close(
            actual=jac_actual, expected=jac_expected, atol=1e-6, rtol=1e-5
        )
