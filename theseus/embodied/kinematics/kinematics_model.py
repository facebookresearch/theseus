# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional, Union

import torch

from theseus.constants import DeviceType
from theseus.geometry import SE3, LieGroup, Point2, Vector
from theseus.labs.embodied.robot import Robot, get_forward_kinematics_fns

RobotModelInput = Union[torch.Tensor, Vector]


class KinematicsModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward_kinematics(
        self,
        robot_pose: RobotModelInput,
        **kwargs,
    ) -> List[LieGroup]:
        pass


class IdentityModel(KinematicsModel):
    def __init__(self):
        super().__init__()

    def forward_kinematics(  # type: ignore
        self, robot_pose: RobotModelInput
    ) -> List[LieGroup]:
        if isinstance(robot_pose, Point2) or isinstance(robot_pose, Vector):
            assert robot_pose.dof() == 2
            return [robot_pose]
        raise NotImplementedError(
            f"IdentityModel not implemented for pose with type {type(robot_pose)}."
        )


class UrdfRobotModel(KinematicsModel):
    def __init__(
        self,
        urdf_path: str,
        device: DeviceType = None,
        dtype: torch.dtype = torch.float32,
        link_names: Optional[List[str]] = None,
    ):
        self.robot = Robot.from_urdf_file(urdf_path, dtype=dtype, device=device)
        self.fk, self.jfk_b, self.jfk_s = get_forward_kinematics_fns(
            self.robot, link_names
        )
        self.link_names = link_names

    def _postprocess_quaternion(self, quat):
        # Convert quaternion convention (DRM uses xyzw, Theseus uses wxyz)
        quat_converted = torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)

        # Normalize quaternions
        quat_normalized = quat_converted / torch.linalg.norm(
            quat_converted, dim=-1, keepdim=True
        )

        return quat_normalized

    def forward_kinematics(  # type: ignore
        self,
        joint_states: RobotModelInput,
        jacobians: Optional[List[torch.Tensor]] = None,
        use_body_jacobians: bool = True,
    ) -> List[SE3]:  # type: ignore
        """Computes forward kinematics for the robot's selected links.

        Args:
            joint_states (tensor or theseus.Vector): Vector of all joint angles
            jacobians (optional): If an empty list is given, it's filled with jacobians
            use_body_jacobian (bool): If true, jacobians are body jacobians, otherwise
                they are spatial jacobians.
        Outputs:
            Dictionary that maps link name to link pose
        """
        if jacobians is not None and len(jacobians) > 0:
            raise ValueError("Jacobians dictionary must be empty on input.")

        # Check input dimensions
        if joint_states.shape[-1] != self.robot.num_joints:
            raise ValueError(
                f"Robot model dofs ({self.robot.num_joints}) incompatible with "
                f"input joint state dimensions ({joint_states.shape[-1]})."
            )

        # Parse input
        if isinstance(joint_states, torch.Tensor):
            joint_states_input = joint_states
        elif isinstance(joint_states, Vector):
            joint_states_input = joint_states.tensor
        else:
            raise Exception(
                "Invalid input joint states data type. "
                "Valid types are torch.Tensor and th.Vector."
            )

        # Compute forward kinematics for all links
        fk_output = self.fk(joint_states_input)

        link_poses = []
        for i, _ in enumerate(self.link_names):
            link_poses.append(SE3(tensor=fk_output[i], strict_checks=False))

        # Compute jacobians
        if jacobians is not None:
            jacs_out = (
                self.jfk_b(joint_states_input)
                if use_body_jacobians
                else self.jfk_s(joint_states_input)
            )
            for i, _ in enumerate(self.link_names):
                jacobians.append(jacs_out[0][i])

        return link_poses
