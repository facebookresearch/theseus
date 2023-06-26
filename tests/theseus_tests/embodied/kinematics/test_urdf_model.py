# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from enum import Enum

import pytest
import torch

import theseus as th

DATA_REL_PATH = "data/panda_fk_dataset.json"
URDF_PATH = os.path.join(os.path.dirname(__file__), "data/panda_no_gripper.urdf")


class VectorType(Enum):
    TORCH_TENSOR = 1
    TH_VECTOR = 2


device = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=[VectorType.TORCH_TENSOR, VectorType.TH_VECTOR])
def dataset(request):
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), DATA_REL_PATH)
    with open(data_path) as f:
        data = json.load(f)

    # Input vector type
    if request.param == VectorType.TORCH_TENSOR:
        joint_states_input = torch.tensor(data["joint_states"], device=device)
    elif request.param == VectorType.TH_VECTOR:
        joint_states_input = th.Vector(
            tensor=torch.tensor(data["joint_states"], device=device)
        )
    else:
        raise Exception("Invalid vector type specified.")

    # Convert ee poses (from xyzw to wxyz, then from list to tensor)
    ee_poses = torch.tensor(
        [pos + quat[3:] + quat[:3] for pos, quat in data["ee_poses"]], device=device
    )

    return {
        "num_data": len(data["joint_states"]),
        "joint_states": joint_states_input,
        "ee_poses": ee_poses,
        "ee_name": data["ee_name"],
    }


def test_forward_kinematics_seq(dataset):
    ee_name = dataset["ee_name"]
    robot_model = th.eb.UrdfRobotModel(URDF_PATH, device=device, link_names=[ee_name])

    for joint_state, ee_pose_target in zip(
        dataset["joint_states"], dataset["ee_poses"]
    ):
        ee_se3_target = th.SE3(x_y_z_quaternion=ee_pose_target)
        ee_se3_computed = robot_model.forward_kinematics(joint_state.view(1, -1))[
            ee_name
        ]

        assert torch.allclose(
            ee_se3_target.local(ee_se3_computed),
            torch.zeros(6, device=device),
            atol=1e-5,
            rtol=1e-4,
        )


def test_forward_kinematics_batched(dataset):
    ee_name = dataset["ee_name"]
    robot_model = th.eb.UrdfRobotModel(URDF_PATH, device=device, link_names=[ee_name])

    ee_se3_target = th.SE3(x_y_z_quaternion=dataset["ee_poses"])
    ee_se3_computed = robot_model.forward_kinematics(dataset["joint_states"])[ee_name]

    assert torch.allclose(
        ee_se3_target.local(ee_se3_computed),
        torch.zeros(dataset["num_data"], 6, device=device),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.fixture
def autograd_jacobians(dataset):
    ee_name = dataset["ee_name"]
    robot_model = th.eb.UrdfRobotModel(URDF_PATH, device=device, link_names=[ee_name])

    jacobians = []
    for joint_state, ee_pose_target in zip(
        dataset["joint_states"], dataset["ee_poses"]
    ):
        ee_se3_target = th.SE3(x_y_z_quaternion=ee_pose_target)

        # Compute autograd manipulator jacobian
        def fk_func(x):
            ee_se3_output = robot_model.forward_kinematics(x)[ee_name]
            delta_pose_ee_frame = ee_se3_target.local(ee_se3_output)
            return delta_pose_ee_frame

        jacobian_autograd = torch.autograd.functional.jacobian(
            fk_func, joint_state.view(-1, 7)
        ).squeeze()

        jacobians.append(jacobian_autograd)

    return torch.stack(jacobians)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_jacobian(dataset, autograd_jacobians, batch_size):
    ee_name = dataset["ee_name"]
    robot_model = th.eb.UrdfRobotModel(URDF_PATH, device=device, link_names=[ee_name])
    joint_state = dataset["joint_states"][0:batch_size, ...].view(batch_size, -1)

    # Compute analytical manipulator jacobian
    jacobians = {}
    robot_model.forward_kinematics(
        joint_state, jacobians=jacobians, use_body_jacobians=True
    )[ee_name]
    jacobian_analytical = jacobians[ee_name]

    assert torch.allclose(
        autograd_jacobians[0:batch_size, ...],
        jacobian_analytical,
        atol=1e-6,
        rtol=1e-3,
    )
