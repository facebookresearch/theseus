import json
import os

import pytest
import torch

import theseus as th
from theseus.geometry import SE3

URDF_REL_PATH = "data/panda_no_gripper.urdf"
DATA_REL_PATH = "data/panda_fk_dataset.json"


@pytest.fixture
def robot_model():
    urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)
    return th.eb.UrdfRobotModel(urdf_path)


@pytest.fixture
def dataset():
    data_path = os.path.join(os.path.dirname(__file__), DATA_REL_PATH)
    with open(data_path) as f:
        data = json.load(f)

    return {
        "num_data": len(data["joint_states"]),
        "joint_states": torch.Tensor(data["joint_states"]),
        "ee_poses": torch.Tensor(
            [pos + quat[3:] + quat[:3] for pos, quat in data["ee_poses"]]
        ),
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
