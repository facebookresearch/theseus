import json
import os
from enum import Enum

import pytest
import torch

import theseus as th
from theseus.geometry import SE3

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
        joint_states_input = th.Vector(data["joint_states"])
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
        ee_se3_target = SE3(x_y_z_quaternion=ee_pose_target)
        ee_se3_computed = robot_model.forward_kinematics(joint_state)[ee_name]

        assert torch.allclose(ee_se3_target.local(ee_se3_computed), torch.zeros(6))


def test_forward_kinematics_batched(robot_model, dataset):
    ee_name = dataset["ee_name"]

    ee_se3_target = SE3(x_y_z_quaternion=dataset["ee_poses"])
    ee_se3_computed = robot_model.forward_kinematics(dataset["joint_states"])[ee_name]

    assert torch.allclose(
        ee_se3_target.local(ee_se3_computed),
        torch.zeros(dataset["num_data"], 6),
    )
