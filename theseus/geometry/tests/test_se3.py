# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS
from theseus.utils import numeric_jacobian

from .common import (
    check_adjoint,
    check_compose,
    check_exp_map,
    check_inverse,
    check_projection_for_compose,
    check_projection_for_inverse,
    check_projection_for_rotate_and_transform,
)


def check_SE3_log_map(tangent_vector, atol=EPS):
    g = th.SE3.exp_map(tangent_vector)
    assert torch.allclose(th.SE3.exp_map(g.log_map()).data, g.data, atol=atol)


def test_exp_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= (
            torch.rand(batch_size, 1, generator=rng).double() * 2 * np.pi - np.pi
        )
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    # SE3.exp_map uses approximations for small theta
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 1e-5
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    # SE3.exp_map uses the exact exponential map for small theta
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 3e-3
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 2 * np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)


def test_log_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng) - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= (
            torch.rand(batch_size, 1, generator=rng).double() * 2 * np.pi - np.pi
        )
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    # SE3.log_map uses approximations for small theta
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 1e-5
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    # SE3.log_map uses the exact logarithm map for small theta
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 3e-3
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 2 * np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3, generator=rng).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3, generator=rng).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)


def test_compose():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3_1 = th.SE3.rand(batch_size, generator=rng, dtype=torch.float64)
        se3_2 = th.SE3.rand(batch_size, generator=rng, dtype=torch.float64)
        check_compose(se3_1, se3_2)


def test_inverse():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3 = th.SE3.rand(batch_size, generator=rng, dtype=torch.float64)
        check_inverse(se3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3 = th.SE3.rand(batch_size, generator=rng, dtype=torch.float64)
        tangent = torch.randn(batch_size, 6).double()
        check_adjoint(se3, tangent)


def test_transform_from_and_to():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in [1, 20, 100]:
            se3 = th.SE3.rand(batch_size, generator=rng, dtype=torch.float64)
            point_tensor = torch.randn(batch_size, 3).double()
            point_tensor_ext = torch.cat(
                (point_tensor, torch.ones(batch_size, 1).double()), dim=1
            )

            jacobians_to = []
            point_to = se3.transform_to(point_tensor, jacobians=jacobians_to)
            expected_to = (se3.to_matrix() @ point_tensor_ext.unsqueeze(2))[:, :3]
            jacobians_from = []
            point_from = se3.transform_from(point_to, jacobians_from)

            # Check the operation result
            assert torch.allclose(expected_to.squeeze(2), point_to.data, atol=EPS)
            assert torch.allclose(point_tensor, point_from.data, atol=EPS)

            # Check the jacobians
            expected_jac = numeric_jacobian(
                lambda groups: groups[0].transform_to(groups[1]),
                [se3, th.Point3(point_tensor)],
                function_dim=3,
            )
            assert torch.allclose(jacobians_to[0], expected_jac[0])
            assert torch.allclose(jacobians_to[1], expected_jac[1])
            expected_jac = numeric_jacobian(
                lambda groups: groups[0].transform_from(groups[1]),
                [se3, point_to],
                delta_mag=1e-5,
                function_dim=3,
            )
            assert torch.allclose(jacobians_from[0], expected_jac[0])
            assert torch.allclose(jacobians_from[1], expected_jac[1])


def test_projection():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in [1, 20, 100]:
            # Test SE3.transform_to
            check_projection_for_rotate_and_transform(
                th.SE3, th.Point3, th.SE3.transform_to, batch_size, rng
            )

            # Test SE3.transform_from
            check_projection_for_rotate_and_transform(
                th.SE3, th.Point3, th.SE3.transform_from, batch_size, rng
            )

            # Test SE3.compose
            check_projection_for_compose(th.SE3, batch_size, rng)

            # Test SE3.inverse
            check_projection_for_inverse(th.SE3, batch_size, rng)
