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

URDF_REL_PATH = "data/panda_no_gripper.urdf"
DATA_REL_PATH = "data/panda_fk_dataset.json"


class VectorType(Enum):
    TORCH_TENSOR = 1
    TH_VECTOR = 2


@pytest.fixture
def robot_model():
    urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)
    return th.eb.UrdfRobotModel(urdf_path)


@pytest.fixture(params=[VectorType.TORCH_TENSOR, VectorType.TH_VECTOR])
def dataset(request):
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), DATA_REL_PATH)
    with open(data_path) as f:
        data = json.load(f)

    # Input vector type
    if request.param == VectorType.TORCH_TENSOR:
        joint_states_input = torch.Tensor(data["joint_states"])
    elif request.param == VectorType.TH_VECTOR:
        joint_states_input = th.Vector(tensor=torch.Tensor(data["joint_states"]))
    else:
        raise Exception("Invalid vector type specified.")

    # Convert ee poses (from xyzw to wxyz, then from list to tensor)
    ee_poses = torch.Tensor(
        [pos + quat[3:] + quat[:3] for pos, quat in data["ee_poses"]]
    )

    return {
        "num_data": len(data["joint_states"]),
        "joint_states": joint_states_input,
        "ee_poses": ee_poses,
        "ee_name": data["ee_name"],
    }


def test_forward_kinematics_seq(robot_model, dataset):
    ee_name = dataset["ee_name"]

    for joint_state, ee_pose_target in zip(
        dataset["joint_states"], dataset["ee_poses"]
    ):
        ee_se3_target = th.SE3(x_y_z_quaternion=ee_pose_target)
        ee_se3_computed = robot_model.forward_kinematics(joint_state)[ee_name]

        assert torch.allclose(
            ee_se3_target.local(ee_se3_computed), torch.zeros(6), atol=1e-5, rtol=1e-4
        )


def test_forward_kinematics_batched(robot_model, dataset):
    ee_name = dataset["ee_name"]

    ee_se3_target = th.SE3(x_y_z_quaternion=dataset["ee_poses"])
    ee_se3_computed = robot_model.forward_kinematics(dataset["joint_states"])[ee_name]

    assert torch.allclose(
        ee_se3_target.local(ee_se3_computed),
        torch.zeros(dataset["num_data"], 6),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.fixture
def autograd_jacobians(robot_model, dataset):
    ee_name = dataset["ee_name"]

    jacobians = []
    for joint_state, ee_pose_target in zip(
        dataset["joint_states"], dataset["ee_poses"]
    ):
        ee_se3_target = th.SE3(x_y_z_quaternion=ee_pose_target)

        # Compute autograd manipulator jacobian
        def fk_func(x):
            ee_se3_output = robot_model.forward_kinematics(x)[ee_name]
            delta_pose_ee_frame = ee_se3_target.local(ee_se3_output)

            adjoint_matrix = ee_se3_target.adjoint()
            adjoint_matrix[:, 0:3, 3:6] = 0.0  # convert only rotational frame
            delta_pose_base_frame = adjoint_matrix @ delta_pose_ee_frame.squeeze()

            return delta_pose_base_frame

        jacobian_autograd = torch.autograd.functional.jacobian(
            fk_func, joint_state
        ).squeeze()

        jacobians.append(jacobian_autograd)

    return torch.stack(jacobians)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_jacobian(robot_model, dataset, autograd_jacobians, batch_size):
    ee_name = dataset["ee_name"]
    joint_state = dataset["joint_states"][0:batch_size, ...]

    # Compute analytical manipulator jacobian
    jac_fk = dict.fromkeys([ee_name])
    robot_model.forward_kinematics(joint_state, jacobians=jac_fk)
    jacobian_analytical = jac_fk[ee_name]

    assert torch.allclose(
        autograd_jacobians[0:batch_size, ...], jacobian_analytical, atol=1e-6, rtol=1e-3
    )
