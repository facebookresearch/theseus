# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Dict, Tuple, Optional
import urdf_parser_py.urdf as urdf
import torch

from theseus.geometry.functional import so3
from .joint import Joint, RevoluteJoint, PrismaticJoint
from .link import Link


class Robot(abc.ABC):
    def __init__(self, name: str, dtype: torch.dtype = None):
        if dtype is None:
            dtype = torch.get_default_dtype()
        self._name: str = name
        self._dtype: torch.dtype = dtype
        self._num_joints: int = 0
        self._num_links: int = 0
        self._joints: List[Joint] = []
        self._links: List[Link] = []
        self._joint_map: Dict[str, Joint] = {}
        self._link_map: Dict[str, Link] = {}

    @classmethod
    def from_urdf_file(urdf_file: str, dtype: torch.dtype = None) -> "Robot":
        if dtype is None:
            dtype = torch.get_default_dtype()

        def get_joint_type(joint: urdf.Joint):
            if joint.type == "revolute" or joint.type == "continuous":
                return RevoluteJoint
            elif joint.type == "prismatic":
                return PrismaticJoint
            else:
                raise ValueError(f"{joint.type} is currently not supported.")

        def get_origin(urdf_origin: Optional[urdf.Pose] = None) -> torch.Tensor:
            origin = torch.eye(3, 4, dtype=dtype).unsqueeze(0)
            if urdf_origin is None:
                return origin

            if urdf_origin.xyz is not None:
                origin[:, :, 3] = torch.Tensor(urdf_origin.xyz, dtype=dtype)

            if urdf_origin.rpy is not None:
                rpy = urdf_origin.rpy
                rot_x = so3.exp(torch.Tensor([[rpy[0], 0, 0]], dtype=dtype))
                rot_y = so3.exp(torch.Tensor([[0, rpy[1], 0]], dtype=dtype))
                rot_z = so3.exp(torch.Tensor([[0, 0, rpy[2]]], dtype=dtype))
                return rot_x @ rot_y @ rot_z

        urdf_model = urdf.URDF.from_xml_file(urdf_file)
        robot = Robot()

        child_joints: Dict[str, List[Joint]] = {}
        for urdf_link in urdf_model.links:
            link = Link(urdf_link.name)
            robot._link_map[urdf_link.name] = link
            child_joints[urdf_link.name] = []

        for urdf_joint in urdf_model.joints:
            joint = Joint(urdf_joint.name)
            robot._joint_map[urdf_joint.name] = joint
            child_joints[urdf_joint.parent].append(joint)

        urdf_link = urdf_model.link_map[urdf_model.get_root()]
        links_to_visit = [(urdf_link, None, None)]

        while links_to_visit:
            urdf_link, joint, origin = links_to_visit.pop(0)
            link = robot.link_map[urdf_link.name]
            if joint is not None:
                link.parent = joint.id

        return robot

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

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
