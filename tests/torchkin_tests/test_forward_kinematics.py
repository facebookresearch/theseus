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

URDF_REL_PATH = "panda_no_gripper.urdf"
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

    def fk_vmap(t):
        return tuple(pose.squeeze(0) for pose in fk(t.unsqueeze(0)))

    rng = torch.Generator()
    rng.manual_seed(0)
    angles = torch.rand(batch_size, robot.dof, generator=rng, dtype=dtype)

    jacs_actual, poses = jfk_b(angles)
    jacs_dense = torch.autograd.functional.jacobian(fk, angles, vectorize=True)
    jacs_vmap = torch.vmap(torch.func.jacrev(fk_vmap))(angles)

    jacs_expected = []
    sels = range(batch_size)
    for pose, jac_dense, jac_vmap in zip(poses, jacs_dense, jacs_vmap):
        jac_sparse = jac_dense[sels, :, :, sels]
        torch.testing.assert_close(
            actual=jac_vmap, expected=jac_sparse, atol=1e-6, rtol=1e-5
        )
        jac_sparse_t = jac_sparse.transpose(-1, 1).transpose(-1, -2)
        jac_expected = SE3.left_project(pose, jac_sparse_t).transpose(-1, -2)
        jac_expected[:, 3:] *= 0.5
        jacs_expected.append(jac_expected)

    for jac_actual, jac_expected in zip(jacs_actual, jacs_expected):
        torch.testing.assert_close(
            actual=jac_actual, expected=jac_expected, atol=1e-6, rtol=1e-5
        )


@pytest.mark.parametrize("batch_size", [1, 20, 40])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vmap_for_jacobian(batch_size: int, dtype: torch.dtype):
    robot = Robot.from_urdf_file(urdf_path, dtype)
    selected_links = ["panda_virtual_ee_link"]
    fk, jfk_b, _ = get_forward_kinematics_fns(robot, selected_links)

    def fun(angles):
        jac_b = jfk_b(angles)[0][0]
        return jac_b.sum(dim=(-1, -2)).unsqueeze(-1)

    def fun_vmap(angles):
        return fun(angles.unsqueeze(0)).squeeze(0)

    rng = torch.Generator()
    rng.manual_seed(0)
    angles = torch.rand(batch_size, robot.dof, generator=rng, dtype=dtype)

    sels = range(batch_size)
    grad = torch.autograd.functional.jacobian(fun, angles, vectorize=True)
    grad_vmap = torch.vmap(torch.func.jacrev(fun_vmap))(angles)

    torch.testing.assert_close(
        actual=grad[sels, :, sels], expected=grad_vmap, atol=1e-6, rtol=1e-5
    )
