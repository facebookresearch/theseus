# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import pytest
import torch

import theseus as th
from theseus.geometry import SE3, SO3

NUM_DOFS = 7
EE_NAME = "panda_virtual_ee_link"
HOME_POSE = [-0.1394, -0.0205, -0.0520, -2.0691, 0.0506, 2.0029, -0.9168]


def sample_vector_dist(range):
    return np.random.choice([-1, 1]) * range * torch.rand(range.shape)


@pytest.fixture
def robot_model():
    urdf_path = os.path.join(os.path.dirname(__file__), "data/panda_no_gripper.urdf")
    return th.eb.UrdfRobotModel(urdf_path)


@pytest.fixture
def ee_pose_target():
    np.random.seed(0)
    torch.manual_seed(0)

    ee_pos_mid = torch.Tensor([0.6, 0.0, 0.5])
    ee_pos_range = torch.Tensor([0.3, 0.5, 0.5])
    ee_quat_mid = torch.Tensor([0.9383, 0.3442, -0.0072, -0.0318])
    ee_quat_range = torch.Tensor([0.5, 0.5, 0.5])

    ee_pos_target = ee_pos_mid + sample_vector_dist(ee_pos_range)
    ee_quat_target = (
        (
            SO3.unit_quaternion_to_SO3(
                ee_quat_mid / torch.linalg.norm(ee_quat_mid)
            ).compose(SO3().exp_map(sample_vector_dist(ee_quat_range).unsqueeze(0)))
        )
        .to_quaternion()
        .squeeze()
    )
    return SE3(
        x_y_z_quaternion=torch.cat([ee_pos_target, ee_quat_target]),
        name="ee_pose_target",
    )


def test_ik_optimization(robot_model, ee_pose_target):
    """Sets up inverse kinematics as an optimization problem that uses forward kinematics"""
    # Initialize variables (joint states)
    theta = th.Vector(NUM_DOFS, name="theta")

    # Define cost (distance between desired and current ee pose)
    def ee_pose_err_fn(optim_vars, aux_vars):
        (theta,) = optim_vars
        (ee_pose_target,) = aux_vars

        ee_pose = robot_model.forward_kinematics(theta.data)[EE_NAME]
        err = ee_pose_target.local(ee_pose)

        return err

    # Set up optimization
    optim_vars = (theta,)
    aux_vars = (ee_pose_target,)

    cost_function = th.AutoDiffCostFunction(
        optim_vars, ee_pose_err_fn, 6, aux_vars=aux_vars, name="ee_pose_err_fn"
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
        "ee_pose_target": ee_pose_target.data,
        "theta": torch.Tensor(HOME_POSE).unsqueeze(0),
    }
    with torch.no_grad():
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
    assert info.last_err < 1e-3
