# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch

import theseus as th

NUM_DOFS = 7
EE_NAME = "panda_virtual_ee_link"
HOME_POSE = torch.Tensor([-0.1394, -0.0205, -0.0520, -2.0691, 0.0506, 2.0029, -0.9168])
ERR_SCALING = torch.Tensor([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])


@pytest.fixture
def robot_model():
    urdf_path = os.path.join(os.path.dirname(__file__), "data/panda_no_gripper.urdf")
    return th.eb.UrdfRobotModel(urdf_path)


@pytest.fixture(params=[1, 3])
def batch_size(request):
    return request.param


@pytest.fixture
def ee_pose_target(batch_size):
    torch.manual_seed(1)

    ee_pos_mid = torch.Tensor([0.6, 0.0, 0.5])
    ee_pos_range = torch.Tensor([0.1, 0.2, 0.2])
    ee_quat_mid = torch.Tensor([0.9383, 0.3442, -0.0072, -0.0318])
    ee_quat_range = torch.Tensor([0.5, 0.5, 0.5])

    ee_pos_dev = ee_pos_range * torch.randn(batch_size, 3)
    ee_rot_dev = ee_quat_range * torch.randn(batch_size, 3)

    ee_pos_target = ee_pos_mid + ee_pos_dev
    ee_quat_target = (
        th.SO3.unit_quaternion_to_SO3(
            ee_quat_mid / torch.linalg.norm(ee_quat_mid)
        ).compose(th.SO3().exp_map(ee_rot_dev))
    ).to_quaternion()
    ee_pose_target = th.SE3(
        x_y_z_quaternion=torch.cat([ee_pos_target, ee_quat_target], dim=-1),
        name="ee_pose_target",
    )

    return ee_pose_target


@pytest.mark.parametrize("is_grad_enabled", [True, False])
def test_ik_optimization(robot_model, batch_size, ee_pose_target, is_grad_enabled):
    """Sets up inverse kinematics as an optimization problem that uses forward kinematics"""
    # Define cost (distance between desired and current ee pose)
    def ee_pose_err_fn(optim_vars, aux_vars):
        (theta,) = optim_vars
        (ee_pose_target,) = aux_vars

        ee_pose = robot_model.forward_kinematics(theta)[EE_NAME]
        pose_err = ee_pose_target.local(ee_pose)

        return pose_err

    # Set up optimization
    optim_vars = (th.Vector(NUM_DOFS, name="theta"),)
    aux_vars = (ee_pose_target,)

    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        ee_pose_err_fn,
        6,
        aux_vars=aux_vars,
        name="ee_pose_err_fn",
        autograd_mode="dense",
    )
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=15,
        step_size=0.5,
    )
    theseus_optim = th.TheseusLayer(optimizer)

    # Optimize
    theseus_inputs = {
        "theta": torch.tile(HOME_POSE.unsqueeze(0), (batch_size, 1)),
        "ee_pose_target": ee_pose_target,
    }

    with torch.set_grad_enabled(is_grad_enabled):
        updated_inputs, info = theseus_optim.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": True,
                "track_error_history": True,
                "damping": 0.1,
            },
        )

    # Check result
    assert torch.allclose(info.best_err, torch.zeros(batch_size), atol=1e-2)
