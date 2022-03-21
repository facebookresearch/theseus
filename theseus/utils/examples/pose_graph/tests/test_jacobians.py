# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th
import theseus.utils.examples.pose_graph as thpg


def test_pose_prior_error():
    for batch_size in [1, 20, 100]:
        aux_id = torch.arange(batch_size)
        pose = th.SE3.rand(batch_size, dtype=torch.float64)
        pose_prior = th.SE3.rand(batch_size, dtype=torch.float64)
        pose_prior_err = thpg.PosePriorError(pose, pose_prior)

        actual = pose_prior_err.jacobians()[0][0]

        def test_fun(g):
            pose_prior_err.pose.update(g)
            return pose_prior_err.error()

        jac_raw = torch.autograd.functional.jacobian(test_fun, (pose.data))[
            aux_id, :, aux_id
        ]
        expected = pose.project(jac_raw, is_sparse=True)
        assert torch.allclose(actual, expected)


def test_relative_pose_error():
    for batch_size in [1, 20, 100]:
        aux_id = torch.arange(batch_size)
        pose1 = th.SE3.rand(batch_size, dtype=torch.float64)
        pose2 = th.SE3.rand(batch_size, dtype=torch.float64)
        relative_pose = th.SE3.rand(batch_size, dtype=torch.float64)
        relative_pose_err = thpg.RelativePoseError(pose1, pose2, relative_pose)

        actual = relative_pose_err.jacobians()[0]

        def test_fun(g1, g2):
            relative_pose_err.pose1.update(g1)
            relative_pose_err.pose2.update(g2)
            return relative_pose_err.error()

        jac_raw = torch.autograd.functional.jacobian(test_fun, (pose1.data, pose2.data))
        expected = [
            pose1.project(jac_raw[0][aux_id, :, aux_id], is_sparse=True),
            pose2.project(jac_raw[1][aux_id, :, aux_id], is_sparse=True),
        ]

        assert torch.allclose(actual[0], expected[0])
        assert torch.allclose(actual[1], expected[1])
