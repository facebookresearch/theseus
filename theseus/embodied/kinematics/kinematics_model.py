# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast

import differentiable_robot_model as drm
import torch
from stl import mesh
from urdf_parser_py.urdf import URDF, Mesh

from theseus.geometry import SE3, LieGroup, Point2, Point3, Vector

RobotModelInput = Union[torch.Tensor, Vector]


@dataclass
class Sphere:
    position: Point3
    radius: float


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
    def __init__(self, urdf_path: str, collision_params: Dict[str, float] = None):
        # Initialize DRM
        self.drm_model = drm.DifferentiableRobotModel(urdf_path)

        # Parse URDF for collision geometries
        self.collision_params = collision_params or {}
        self.collision_spheres = {}

        robot = URDF.from_xml_file(urdf_path)
        for link in robot.links:
            if link.collision is not None and type(link.collision.geometry) is Mesh:
                # Load mesh file
                mesh_path = os.path.join(
                    os.path.dirname(urdf_path), link.collision.geometry.filename
                )
                mesh_obj = mesh.Mesh.from_file(mesh_path)

                # Process mesh
                self.collision_spheres[link.name] = self._generate_spheres_from_mesh(
                    mesh_obj
                )

    def _generate_spheres_from_mesh(self, mesh):
        """Approximates a mesh with a collection of spheres

        Current placeholder primitive implementation: Generate a single sphere
        located at the COM of the mesh, with r = distance to farthest triangle
        """
        # Find center of each triangle
        mesh_coms = (
            torch.Tensor(
                mesh.points[:, 0:3] + mesh.points[:, 3:6] + mesh.points[:, 6:9]
            )
            / 3.0
        )

        # Sphere center as COM of all COMs
        center = torch.mean(mesh_coms, dim=0)

        # Sphere radius as farthest point from center
        radius = float(torch.max(torch.linalg.norm(mesh_coms - center, dim=-1)))

        return [Sphere(position=Point3(center), radius=radius)]

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
            joint_states_input = joint_states.data
        else:
            raise Exception(
                "Invalid input joint states data type. Valid types are torch.Tensor and th.Vector."
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

    def get_collision_spheres(self, link_states: Dict[str, LieGroup]) -> List[Sphere]:
        spheres_ret = []
        for link_name in link_states:
            # Skip link if no collision spheres are associated
            if link_name not in self.collision_spheres:
                continue

            # Apply link pose to link spheres
            link_transform = link_states[link_name]
            for sphere in self.collision_spheres[link_name]:
                assert isinstance(
                    link_transform, SE3
                ), f'Input link states must be "th.SE3", instead got "{type(link_transform)}".'

                sphere_pos_transformed = cast(SE3, link_transform).transform_from(
                    sphere.position
                )
                spheres_ret.append(
                    Sphere(
                        position=sphere_pos_transformed,
                        radius=sphere.radius,
                    )
                )

        return spheres_ret
