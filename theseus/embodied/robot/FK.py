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

    ancestors = []

    for link in links:
        ancestors += [anc.id for anc in link.ancestors]

    link_ids = sorted(list(set(ancestors)))

    def ForwardKinematicsImpl(angles: torch.Tensor):
        if angles.ndim != 2 or angles.shape[1] != robot.dof:
            raise ValueError(
                f"Joint angles for {robot.name} should be {robot.dof}-D vectors"
            )

        poses: List[Optional[torch.Tensor]] = [None] * robot.num_links
        poses[0] = angles.new_zeros(angles.shape[0], 3, 4)
        poses[0][:, 0, 0] = 1
        poses[0][:, 1, 1] = 1
        poses[0][:, 2, 2] = 1

        for id in link_ids[1:]:
            curr: Link = robot.links[id]
            joint: Joint = robot.links[id].parent
            prev: Link = joint.parent
            relative_pose = (
                joint.relative_pose(angles[:, joint.id])
                if joint.id < robot.dof
                else joint.relative_pose()
            )
            poses[curr.id] = se3.compose(poses[prev.id], relative_pose)

        return tuple(poses[id] for id in link_ids)

    return ForwardKinematicsImpl
