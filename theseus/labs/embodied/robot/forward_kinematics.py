# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import List, Optional

from lie.functional import SE3
from .robot import Robot, Joint, Link

# TODO: Add support for joints with DOF>1


def ForwardKinematicsFactory(robot: Robot, link_names: Optional[List[str]] = None):
    links: List[Link] = robot.get_links(link_names)
    link_ids: List[int] = [link.id for link in links]

    ancestor_links: List[Link] = []
    joint_ids: List[int] = []
    for link in links:
        ancestor_links += [anc for anc in link.ancestor_links]
        joint_ids += link.ancestor_non_fixed_joint_ids
    pose_ids = sorted(list(set([anc.id for anc in ancestor_links] + link_ids)))
    joint_ids = sorted(list(set(joint_ids)))

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
            joint: Joint = robot.links[id].parent_joint
            prev: Link = joint.parent_link
            relative_pose = (
                joint.relative_pose(angles[:, joint.id])
                if joint.id < robot.dof
                else joint.relative_pose()
            )
            poses[curr.id] = SE3.compose(poses[prev.id], relative_pose)

        return poses

    def _forward_kinematics_impl(angles: torch.Tensor):
        poses = _forward_kinematics_helper(angles)
        return tuple(poses[id] for id in link_ids)

    def _jforward_kinematics_helper(
        poses: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        jposes = poses[0].new_zeros(poses[0].shape[0], 6, robot.dof)

        for id in pose_ids[1:]:
            link: Link = robot.links[id]
            joint: Joint = link.parent_joint
            if joint.id >= robot.dof:
                break
            jposes[:, :, joint.id : joint.id + 1] = SE3.adj(poses[link.id]) @ joint.axis

        return jposes

    def _jforward_kinematics_impl(angles: torch.Tensor):
        poses = _forward_kinematics_helper(angles)
        jposes = _jforward_kinematics_helper(poses)

        rets = tuple(poses[id] for id in link_ids)
        jacs: List[torch.Tensor] = []

        for link_id in link_ids:
            pose = poses[link_id]
            jac = jposes.new_zeros(angles.shape[0], 6, robot.dof)
            sel = robot.links[link_id].ancestor_non_fixed_joint_ids
            jac[:, :, sel] = SE3.adj(SE3.inv(pose)) @ jposes[:, :, sel]
            jacs.append(jac)

        return jacs, rets

    class ForwardKinematics(torch.autograd.Function):
        @classmethod
        def forward(cls, ctx, angles):
            poses = _forward_kinematics_helper(angles)
            ctx.poses = poses
            rets = tuple(poses[id] for id in link_ids)
            ctx.rets = rets
            return rets

        @classmethod
        def backward(cls, ctx, *grad_outputs):
            if not hasattr(ctx, "jposes"):
                ctx.jposes: torch.Tensor = _jforward_kinematics_helper(ctx.poses)
            rets: tuple(torch.Tensor) = ctx.rets
            grad_pose = grad_outputs[0].new_zeros(
                grad_outputs[0].shape[0], 6, robot.dof
            )
            grad_input = grad_outputs[0].new_zeros(grad_outputs[0].shape[0], robot.dof)

            for link_id, ret, grad_output in zip(link_ids, rets, grad_outputs):
                ancestor_non_fixed_joint_ids = robot.links[
                    link_id
                ].ancestor_non_fixed_joint_ids
                temp = SE3.project(
                    torch.cat(
                        (grad_output @ ret.transpose(1, 2), grad_output[:, :, 3:]),
                        dim=-1,
                    )
                ).unsqueeze(-1)
                grad_pose[:, :, ancestor_non_fixed_joint_ids] += temp
            grad_input[:, joint_ids] = (
                ctx.jposes[:, :, joint_ids] * grad_pose[:, :, joint_ids]
            ).sum(-2)

            return grad_input

    return (
        ForwardKinematics,
        _forward_kinematics_impl,
        _jforward_kinematics_impl,
        _forward_kinematics_helper,
        _jforward_kinematics_helper,
    )


def get_forward_kinematics(robot: Robot, link_names: Optional[List[str]] = None):
    ForwardKinematics, _, jforward_kinematics, _, _ = ForwardKinematicsFactory(
        robot, link_names
    )
    forward_kinematics = ForwardKinematics.apply
    return forward_kinematics, jforward_kinematics
