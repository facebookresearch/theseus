# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import List, Optional

from theseus.geometry.functional import se3
from .robot import Robot
from .link import Link
from .joint import Joint

# TODO: Add support for joints with DOF>1


def ForwardKinematicFactory(robot: Robot, link_names: Optional[List[str]] = None):
    links: List[Link] = []

    if link_names is None:
        links = robot.links
    else:
        links = [robot.link_map[name] for name in link_names]

    link_ids: List[int] = [link.id for link in links]

    ancestors = []
    for link in links:
        ancestors += [anc for anc in link.ancestors]
    pose_ids = sorted(list(set([anc.id for anc in ancestors] + link_ids)))

    def _forward_kinematics_helper(angles: torch.Tensor):
        if angles.ndim != 2 or angles.shape[1] != robot.dof:
            raise ValueError(
                f"Joint angles for {robot.name} should be {robot.dof}-D vectors"
            )

        poses: List[Optional[torch.Tensor]] = [None] * robot.num_links
        poses[0] = angles.new_zeros(angles.shape[0], 3, 4)
        poses[0][:, 0, 0] = 1
        poses[0][:, 1, 1] = 1
        poses[0][:, 2, 2] = 1

        for id in pose_ids[1:]:
            curr: Link = robot.links[id]
            joint: Joint = robot.links[id].parent
            prev: Link = joint.parent
            relative_pose = (
                joint.relative_pose(angles[:, joint.id])
                if joint.id < robot.dof
                else joint.relative_pose()
            )
            poses[curr.id] = se3.compose(poses[prev.id], relative_pose)

        return poses

    def _forward_kinematics_impl(angles: torch.Tensor):
        poses = _forward_kinematics_helper(angles)
        return tuple(poses[id] for id in link_ids)

    def _jforward_kinematics_helper(poses: List[Optional[torch.Tensor]]):
        jposes = poses[0].new_zeros(poses[0].shape[0], 6, robot.dof)

        for id in pose_ids[1:]:
            joint: Joint = robot.links[id].parent
            prev: Link = joint.parent
            if joint.id >= robot.dof:
                break
            jposes[:, :, joint.id : joint.id + 1] = (
                se3.adjoint(poses[prev.id]) @ joint.axis
            )

        return jposes

    def _jforward_kinematics_impl(angles: torch.Tensor):
        poses = _forward_kinematics_helper(angles)
        jposes = _jforward_kinematics_helper(poses)

        rets = tuple(poses[id] for id in link_ids)
        jacs: List[torch.Tensor] = []

        for link_id in link_ids:
            pose = poses[link_id]
            jac = jposes.new_zeros(angles.shape[0], 6, robot.dof)
            sel = robot.links[link_id].angle_ids
            jac[:, :, sel] = se3.adjoint(se3.inverse(pose)) @ jposes[:, :, sel]
            jacs.append(jac)

        return jacs, rets

    return _forward_kinematics_impl, _jforward_kinematics_impl
