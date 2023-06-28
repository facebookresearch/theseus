# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from torchlie.functional import SE3

from .robot import Joint, Link, Robot

# TODO: Add support for joints with DOF>1


def ForwardKinematicsFactory(robot: Robot, link_names: Optional[List[str]] = None):
    links: List[Link] = robot.get_links(link_names)
    link_ids: List[int] = [link.id for link in links]

    ancestor_links: List[Link] = []
    joint_ids: List[int] = []
    for link in links:
        ancestor_links += [anc for anc in link.ancestor_links]
        joint_ids += link.ancestor_non_fixed_joint_ids
    related_link_ids = sorted(list(set([anc.id for anc in ancestor_links] + link_ids)))
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

        for id in related_link_ids[1:]:
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

    def _forward_kinematics_backward_helper(
        poses: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        jposes = poses[0].new_zeros(poses[0].shape[0], 6, robot.dof)

        for id in related_link_ids[1:]:
            link: Link = robot.links[id]
            joint: Joint = link.parent_joint
            if joint.id >= robot.dof:
                break
            jposes[:, :, joint.id : joint.id + 1] = SE3.adj(poses[link.id]) @ joint.axis

        return jposes

    class ForwardKinematics(torch.autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def forward(angles):
            poses = _forward_kinematics_helper(angles)
            rets = tuple(poses[id] for id in link_ids)
            return *rets, poses

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*output[:-1])
            ctx.poses = output[-1]

        @staticmethod
        def backward(ctx, *grad_outputs):
            rets: tuple(torch.Tensor) = ctx.saved_tensors
            if not hasattr(ctx, "jposes"):
                ctx.jposes: torch.Tensor = _forward_kinematics_backward_helper(
                    ctx.poses
                )
            grad_pose = grad_outputs[-2].new_zeros(
                grad_outputs[-2].shape[0], 6, robot.dof
            )
            grad_input = grad_outputs[-2].new_zeros(
                grad_outputs[-2].shape[0], robot.dof
            )

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
        _forward_kinematics_helper,
        _forward_kinematics_backward_helper,
        link_ids,
        joint_ids,
        related_link_ids,
    )


def get_forward_kinematics_fns(robot: Robot, link_names: Optional[List[str]] = None):
    (
        ForwardKinematics,
        _,
        _,
        backward_helper,
        link_ids,
        _,
        related_link_ids,
    ) = ForwardKinematicsFactory(robot, link_names)

    links = robot.get_links()
    selected_link_names = [links[id].name for id in related_link_ids]
    SelectedForwardKinematics, *_ = ForwardKinematicsFactory(robot, selected_link_names)

    def forward_kinematics(angles: torch.Tensor):
        output = ForwardKinematics.apply(angles)
        return output[:-1]

    def forward_kinematics_spatial_jacobian(angles: torch.Tensor):
        selected_poses: torch.Tensor = SelectedForwardKinematics.apply(angles)
        poses = [None] * robot.num_links
        for id, pose in zip(related_link_ids, selected_poses):
            poses[id] = pose
        jposes: torch.Tensor = backward_helper(poses)

        rets = tuple(poses[id] for id in link_ids)
        jacs_s: List[torch.Tensor] = []

        for link_id in link_ids:
            pose = poses[link_id]
            jac_s = jposes.new_zeros(angles.shape[0], 6, robot.dof)
            sel = robot.links[link_id].ancestor_non_fixed_joint_ids
            jac_s[:, :, sel] = jposes[:, :, sel]
            jacs_s.append(jac_s)

        return jacs_s, rets

    def forward_kinematics_body_jacobian(angles: torch.Tensor):
        jacs_s, rets = forward_kinematics_spatial_jacobian(angles)
        jacs_b: List[torch.Tensor] = []

        for jac_s, pose in zip(jacs_s, rets):
            jacs_b.append(SE3.adj(SE3.inv(pose)) @ jac_s)

        return jacs_b, rets

    return (
        forward_kinematics,
        forward_kinematics_body_jacobian,
        forward_kinematics_spatial_jacobian,
    )
