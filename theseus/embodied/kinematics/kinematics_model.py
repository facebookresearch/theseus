# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict, Optional, Union

import torch

from theseus.constants import DeviceType
from theseus.geometry import SE3, LieGroup, Point2, Vector

RobotModelInput = Union[torch.Tensor, Vector]


class KinematicsModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward_kinematics(self, robot_pose: RobotModelInput) -> Dict[str, LieGroup]:
        pass


class IdentityModel(KinematicsModel):
    def __init__(self):
        super().__init__()

    def forward_kinematics(self, robot_pose: RobotModelInput) -> Dict[str, LieGroup]:
        if isinstance(robot_pose, Point2) or isinstance(robot_pose, Vector):
            assert robot_pose.dof() == 2
            return {"state": robot_pose}
        raise NotImplementedError(
            f"IdentityModel not implemented for pose with type {type(robot_pose)}."
        )


class UrdfRobotModel(KinematicsModel):
    def __init__(self, urdf_path: str, device: DeviceType = None):
        try:
            import differentiable_robot_model as drm
        except ModuleNotFoundError as e:
            print(
                "UrdfRobotModel requires installing differentiable-robot-model. "
                "Please run `pip install differentiable-robot-model`."
            )
            raise e

        self.drm_model = drm.DifferentiableRobotModel(urdf_path, device=device)

    def _postprocess_quaternion(self, quat):
        # Convert quaternion convention (DRM uses xyzw, Theseus uses wxyz)
        quat_converted = torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)

        # Normalize quaternions
        quat_normalized = quat_converted / torch.linalg.norm(
            quat_converted, dim=-1, keepdim=True
        )

        return quat_normalized

    def forward_kinematics(
        self,
        joint_states: RobotModelInput,
        jacobians: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> Dict[str, LieGroup]:
        """Computes forward kinematics
        Args:
            joint_states: Vector of all joint angles
        Outputs:
            Dictionary that maps link name to link pose
        """
        # Check input dimensions
        robot_model_dofs = len(self.drm_model.get_joint_limits())
        assert joint_states.shape[-1] == robot_model_dofs, (
            f"Robot model dofs ({robot_model_dofs}) incompatible with "
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
        fk_output = self.drm_model.compute_forward_kinematics_all_links(
            joint_states_input
        )

        link_poses: Dict[str, LieGroup] = {}
        for link_name in self.drm_model.get_link_names():
            pos, quat = fk_output[link_name]
            quat_processed = self._postprocess_quaternion(quat)

            link_poses[link_name] = SE3(
                x_y_z_quaternion=torch.cat([pos, quat_processed], dim=-1)
            )

        # Compute jacobians
        if jacobians is not None:
            for link_name in jacobians.keys():
                jac_lin, jac_rot = self.drm_model.compute_endeffector_jacobian(
                    joint_states_input, link_name
                )
                jacobians[link_name] = torch.cat([jac_lin, jac_rot], dim=-2)

        return link_poses
