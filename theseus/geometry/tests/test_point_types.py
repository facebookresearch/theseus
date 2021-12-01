# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th


def test_xy_point2():
    for _ in range(100):
        for batch_size in [1, 10, 100]:
            point = th.Point2(data=torch.randn(batch_size, 2))
            assert point.x().allclose(point.data[:, 0])
            assert point.y().allclose(point.data[:, 1])


def test_xyz_point3():
    for _ in range(100):
        for batch_size in [1, 10, 100]:
            point = th.Point3(data=torch.randn(batch_size, 3))
            assert point.x().allclose(point.data[:, 0])
            assert point.y().allclose(point.data[:, 1])
            assert point.z().allclose(point.data[:, 2])


def test_point_operations_return_correct_type():
    for point_cls in [th.Point2, th.Point3]:
        p1 = point_cls()
        p2 = point_cls()

        assert isinstance(p1 + p2, point_cls)
        assert isinstance(p1 - p2, point_cls)
        assert isinstance(p1 * p2, point_cls)
        assert isinstance(p1 / p2, point_cls)
        assert isinstance(p1.abs(), point_cls)
        assert isinstance(-p1, point_cls)
        assert isinstance(p1.compose(p2), point_cls)
        assert isinstance(p1.retract(p2.data), point_cls)

        # for these, test result also since this method was overridden
        p1_copy = p1.copy()
        assert isinstance(p1_copy, point_cls)
        assert p1_copy.allclose(p1)
        exp_map = point_cls.exp_map(p2.data)
        assert isinstance(exp_map, point_cls)
        assert exp_map.allclose(p2)
