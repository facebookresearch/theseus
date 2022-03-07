# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.core.tests.common import check_copy_var
from theseus.utils import numeric_jacobian

from .common import (
    check_adjoint,
    check_compose,
    check_exp_map,
    check_inverse,
    check_log_map,
    check_projection_for_compose,
    check_projection_for_inverse,
    check_projection_for_rotate_and_transform,
)


def create_random_se2(batch_size, rng):
    theta = torch.rand(batch_size, 1, generator=rng) * 2 * np.pi - np.pi
    u = torch.randn(batch_size, 2)
    tangent_vector = torch.cat([u, theta], dim=1)
    return th.SE2.exp_map(tangent_vector.double())


def test_exp_map():
    for batch_size in [1, 20, 100]:
        theta = torch.from_numpy(np.linspace(-np.pi, np.pi, batch_size))
        u = torch.randn(batch_size, 2)
        tangent_vector = torch.cat([u, theta.unsqueeze(1)], dim=1)
        check_exp_map(tangent_vector.double(), th.SE2)


def test_log_map():
    for batch_size in [1, 20, 100]:
        theta = torch.from_numpy(np.linspace(-np.pi, np.pi, batch_size))
        u = torch.randn(batch_size, 2)
        tangent_vector = torch.cat([u, theta.unsqueeze(1)], dim=1)
        check_log_map(tangent_vector, th.SE2)


def test_compose():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se2_1 = th.SE2.rand(batch_size, generator=rng, dtype=torch.float64)
        se2_2 = th.SE2.rand(batch_size, generator=rng, dtype=torch.float64)
        check_compose(se2_1, se2_2)


def test_inverse():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=torch.float64)
        check_inverse(se2)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=torch.float64)
        tangent = torch.randn(batch_size, 3).double()
        check_adjoint(se2, tangent)


def test_copy():
    rng = torch.Generator()
    se2 = th.SE2.rand(1, generator=rng, dtype=torch.float64)
    check_copy_var(se2)


def test_transform_to():
    pose = th.SE2(x_y_theta=torch.DoubleTensor([1, 2, np.pi / 2.0]).view(1, -1))
    point = th.Point2(data=torch.DoubleTensor([-1, 4]).view(1, -1))
    expected = th.Point2(data=torch.DoubleTensor([2, 2]).view(1, -1))
    jacobians = []
    actual = pose.transform_to(point, jacobians=jacobians)
    jacobians_from_tensor = []
    actual_from_tensor = pose.transform_to(point.data, jacobians=jacobians_from_tensor)

    expected_jac = numeric_jacobian(
        lambda groups: groups[0].transform_to(groups[1]),
        [pose, point],
        function_dim=2,
    )
    assert expected.data.allclose(actual.data)
    assert expected.data.allclose(actual_from_tensor.data)
    for i in range(2):
        assert torch.allclose(jacobians[i], expected_jac[i])
        assert torch.allclose(jacobians_from_tensor[i], expected_jac[i])


def test_xy_jacobian():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=torch.float64)
        jacobian = []
        se2.xy(jacobians=jacobian)
        expected_jac = numeric_jacobian(
            lambda groups: th.Point2(groups[0].xy()), [se2], function_dim=2
        )
        torch.allclose(jacobian[0], expected_jac[0])


def test_theta_jacobian():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=torch.float64)
        jacobian = []
        se2.theta(jacobians=jacobian)
        expected_jac = numeric_jacobian(
            lambda groups: th.Vector(data=groups[0].theta()), [se2], function_dim=1
        )
        torch.allclose(jacobian[0], expected_jac[0])


def test_projection():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in [1, 20, 100]:
            # Test SE2.transform_to
            check_projection_for_rotate_and_transform(
                th.SE2, th.Point2, th.SE2.transform_to, batch_size, rng
            )

            # Test SE2.compose
            check_projection_for_compose(th.SE2, batch_size, rng)

            # Test SE2.inverse
            check_projection_for_inverse(th.SE2, batch_size, rng)
