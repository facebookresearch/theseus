# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS

from .common import (
    BATCH_SIZES_TO_TEST,
    check_jacobian_for_local,
    check_projection_for_exp_map,
    check_projection_for_log_map,
)


def test_xy_point2():
    for _ in range(100):
        for batch_size in BATCH_SIZES_TO_TEST:
            point = th.Point2(tensor=torch.randn(batch_size, 2))
            assert point.x().allclose(point.tensor[:, 0])
            assert point.y().allclose(point.tensor[:, 1])


def test_xyz_point3():
    for _ in range(100):
        for batch_size in BATCH_SIZES_TO_TEST:
            point = th.Point3(tensor=torch.randn(batch_size, 3))
            assert point.x().allclose(point.tensor[:, 0])
            assert point.y().allclose(point.tensor[:, 1])
            assert point.z().allclose(point.tensor[:, 2])


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
        assert isinstance(p1.retract(p2.tensor), point_cls)

        # for these, test result also since this method was overridden
        p1_copy = p1.copy()
        assert isinstance(p1_copy, point_cls)
        assert p1_copy.allclose(p1)
        exp_map = point_cls.exp_map(p2.tensor)
        assert isinstance(exp_map, point_cls)
        assert exp_map.allclose(p2)


def test_operations_mypy_cast():
    # mypy is optional install, only needed for library contributors
    try:
        import mypy.api
    except ModuleNotFoundError:
        return
    result = mypy.api.run(["tests/geometry/point_types_mypy_check.py"])
    assert result[2] == 0


def test_exp_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in BATCH_SIZES_TO_TEST:
        tangent_vector = torch.rand(batch_size, 2, generator=rng).double() - 0.5
        ret = th.Point2.exp_map(tangent_vector)

        assert torch.allclose(ret.tensor, tangent_vector, atol=EPS)
        check_projection_for_exp_map(
            tangent_vector, Group=th.Point2, is_projected=False
        )

    for batch_size in BATCH_SIZES_TO_TEST:
        tangent_vector = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        ret = th.Point3.exp_map(tangent_vector)

        assert torch.allclose(ret.tensor, tangent_vector, atol=EPS)
        check_projection_for_exp_map(
            tangent_vector, Group=th.Point3, is_projected=False
        )


def test_log_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in BATCH_SIZES_TO_TEST:
        group = th.Point2.rand(batch_size)
        ret = group.log_map()

        assert torch.allclose(ret, group.tensor, atol=EPS)
        check_projection_for_log_map(
            tangent_vector=ret, Group=th.Point2, is_projected=False
        )

    for batch_size in BATCH_SIZES_TO_TEST:
        group = th.Point3.rand(batch_size)
        ret = group.log_map()

        assert torch.allclose(ret, group.tensor, atol=EPS)
        check_projection_for_log_map(
            tangent_vector=ret, Group=th.Point3, is_projected=False
        )


def test_local_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in BATCH_SIZES_TO_TEST:
        group0 = th.Point2.rand(batch_size)
        group1 = th.Point2.rand(batch_size)

        check_jacobian_for_local(group0, group1, Group=th.Point2, is_projected=False)

    for batch_size in BATCH_SIZES_TO_TEST:
        group0 = th.Point3.rand(batch_size)
        group1 = th.Point3.rand(batch_size)

        check_jacobian_for_local(group0, group1, Group=th.Point3, is_projected=False)
