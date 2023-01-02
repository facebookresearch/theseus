# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Dict, Optional
import urdf_parser_py.urdf as urdf
import torch

from theseus.geometry.functional import so3, se3
from .joint import Joint, FixedJoint, RevoluteJoint, PrismaticJoint
from .link import Link


class Robot(abc.ABC):
    def __init__(self, name: str, dtype: torch.dtype = None):
        if dtype is None:
            dtype = torch.get_default_dtype()
        self._name: str = name
        self._dtype: torch.dtype = dtype
        self._dof: int = 0
        self._num_joints: int = 0
        self._num_links: int = 0
        self._joints: List[Joint] = []
        self._links: List[Link] = []
        self._joint_map: Dict[str, Joint] = {}
        self._link_map: Dict[str, Link] = {}

    @classmethod
    def from_urdf_file(cls, urdf_file: str, dtype: torch.dtype = None) -> "Robot":
        if dtype is None:
            dtype = torch.get_default_dtype()

        def get_joint_type(joint: urdf.Joint):
            if joint.type == "revolute" or joint.type == "continuous":
                return RevoluteJoint
            elif joint.type == "prismatic":
                return PrismaticJoint
            elif joint.type == "fixed":
                return FixedJoint
            else:
                raise ValueError(f"{joint.type} is currently not supported.")

        def get_origin(urdf_origin: Optional[urdf.Pose] = None) -> torch.Tensor:
            origin = torch.eye(3, 4, dtype=dtype).unsqueeze(0)
            if urdf_origin is None:
                return origin

            if urdf_origin.xyz is not None:
                origin[:, :, 3] = torch.tensor(urdf_origin.xyz, dtype=dtype)

            if urdf_origin.rpy is not None:
                rpy = urdf_origin.rpy
                rot_x = so3.exp(torch.tensor([[rpy[0], 0, 0]], dtype=dtype))
                rot_y = so3.exp(torch.tensor([[0, rpy[1], 0]], dtype=dtype))
                rot_z = so3.exp(torch.tensor([[0, 0, rpy[2]]], dtype=dtype))
                origin = rot_x @ rot_y @ rot_z

            return origin

        urdf_model = urdf.URDF.from_xml_file(urdf_file)
        robot = Robot(urdf_model.name, dtype)

        for urdf_link in urdf_model.links:
            link = Link(urdf_link.name)
            robot._link_map[urdf_link.name] = link

        for urdf_joint in urdf_model.joints:
            origin = get_origin(urdf_joint.origin)
            joint_type = get_joint_type(urdf_joint.type)
            parent = robot.link_map[urdf_joint.parent]
            child = robot.link_map[urdf_joint.child]
            if joint_type == RevoluteJoint or joint_type == PrismaticJoint:
                axis = torch.tensor(urdf_joint.axis, dtype=dtype)
                joint = joint_type(
                    urdf_joint.name,
                    axis,
                    parent=parent,
                    child=child,
                    origin=origin,
                )
            else:
                joint = joint_type(
                    urdf_joint.name, parent=parent, child=child, origin=origin
                )
            child.set_parent(joint)
            parent.add_child(joint)
            robot._joint_map[urdf_joint.name] = joint

        for _, link in robot.link_map.items():
            for joint in link._children:
                if isinstance(joint, FixedJoint):
                    link.remove_child(joint)
                    subjoints: List[Joint] = joint.child.children
                    joint.child.set_children([])
                    for subjoint in subjoints:
                        subjoint.set_parent(link)
                        subjoint.set_origin(se3.compose(joint.origin, subjoint.origin))
                        link.children.append(subjoint)

        joints_to_visit: List[Joint] = []
        root = robot.link_map[urdf_model.get_root()]
        num_joints = 0
        root.set_id(0)
        robot._links.append(root)
        joints_to_visit = root.children

        for joint in joints_to_visit:
            if not isinstance(joint, FixedJoint):
                joint.set_id(num_joints)
                robot._joints.append(joint)
                num_joints = num_joints + 1
                joint.child.set_id(num_joints)
                robot._links.append(joint.child)

                joints_to_visit = joints_to_visit + joint.child.children

        robot._dof = num_joints

        for _, joint in robot.joint_map.items():
            if joint.id >= 0:
                continue
            if not isinstance(joint.parent, FixedJoint):
                raise ValueError(f"{joint.name} is expected to a fixed joint.")
            joint.set_id(num_joints)
            robot._joints.append(joint)
            num_joints = num_joints + 1
            joint.child.set_id(num_joints)
            robot._links.append(joint.child)

        robot._num_links = len(robot.links)
        robot._num_joints = len(robot.joints)

        return robot

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def num_links(self) -> int:
        return self._num_links

    @property
    def joints(self) -> List[Joint]:
        return self._joints

    @property
    def links(self) -> List[Link]:
        return self._links

    @property
    def joint_map(self) -> Dict[str, Joint]:
        return self._joint_map

    @property
    def link_map(self) -> Dict[str, Link]:
        return self._link_map
