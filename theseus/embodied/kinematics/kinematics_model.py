# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Dict, List, Optional, Union

import torch

from theseus.constants import DeviceType
from theseus.geometry import SE3, LieGroup, Point2, Vector

# The type check below passes when running mypy but not when running pre-commit
# Life is too short to spend hours figuring out how to tell pre-commit to do the
# right thing.
from torchkin import Robot, get_forward_kinematics_fns  # type: ignore

RobotModelInput = Union[torch.Tensor, Vector]


class KinematicsModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward_kinematics(
        self,
        robot_pose: RobotModelInput,
        **kwargs,
    ) -> Dict[str, LieGroup]:
        pass


class IdentityModel(KinematicsModel):
    def __init__(self):
        super().__init__()

    def forward_kinematics(  # type: ignore
        self, robot_pose: RobotModelInput
    ) -> Dict[str, LieGroup]:
        if isinstance(robot_pose, Point2) or isinstance(robot_pose, Vector):
            assert robot_pose.dof() == 2
            return {"state": robot_pose}
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

    def forward_kinematics(  # type: ignore
        self,
        joint_states: RobotModelInput,
        jacobians: Optional[Dict[str, torch.Tensor]] = None,
        use_body_jacobians: bool = True,
    ) -> Dict[str, SE3]:  # type: ignore
        """Computes forward kinematics for the robot's selected links.

        Args:
            joint_states (tensor or theseus.Vector): Vector of all joint angles
            jacobians (dict[str, tensor], optional): If an empty dict is given,
                it's filled with jacobians mapped to their link names.
            use_body_jacobian (bool): If true, jacobians are body jacobians, otherwise
                they are spatial jacobians.
        Outputs:
            Dictionary that maps link name to link pose.
        """
        if jacobians is not None and len(jacobians) > 0:
            raise ValueError("Jacobians dictionary must be empty on input.")

        # Check input dimensions
        if joint_states.shape[-1] != self.robot.dof:
            raise ValueError(
                f"Robot model dofs ({self.robot.dof}) incompatible with "
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

        # Compute jacobians
        poses_list = []
        if jacobians is not None:
            warnings.warn(
                "As of v0.2.0, the kinematics jacobian has changed, and we no longer "
                "rotate the body jacobian to be aligned with the base. Instead, "
                "we compute the classical body jacobian, where each jacobian is in "
                "the local frame of its corresponding link. To get spatial jacobians "
                "completely aligned (translation/rotation) with the base, "
                "set `use_body_jacobian=False.",
                UserWarning,
            )
            jfk_fn = self.jfk_b if use_body_jacobians else self.jfk_s
            jac_links, poses_list = jfk_fn(joint_states_input)
            for i, name in enumerate(self.link_names):
                jacobians[name] = jac_links[i]

        # Do forward kinematics for all links (if not done by jacobians function)
        poses_list = poses_list or self.fk(joint_states_input)

        link_poses = {}
        for i, name in enumerate(self.link_names):
            link_poses[name] = SE3(tensor=poses_list[i], strict_checks=False)

        return link_poses
