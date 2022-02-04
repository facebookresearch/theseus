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

from .common import check_adjoint, check_compose, check_exp_map, check_inverse


def _create_random_se3(batch_size, rng):
    tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
    tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
    tangent_vector_ang *= torch.rand(batch_size, 1).double() * 2 * np.pi - np.pi
    tangent_vector_lin = torch.randn(batch_size, 3).double()
    tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)
    return th.SE3.exp_map(tangent_vector)


def check_SE3_log_map(tangent_vector, atol=EPS):
    g = th.SE3.exp_map(tangent_vector)
    assert torch.allclose(th.SE3.exp_map(g.log_map()).data, g.data, atol=atol)


def test_exp_map():
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= torch.rand(batch_size, 1).double() * 2 * np.pi - np.pi
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 1e-5
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 3e-3
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 2 * np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)


def test_log_map():
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3) - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= torch.rand(batch_size, 1).double() * 2 * np.pi - np.pi
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 1e-5
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 3e-3
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 2 * np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)


def test_compose():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3_1 = _create_random_se3(batch_size, rng)
        se3_2 = _create_random_se3(batch_size, rng)
        check_compose(se3_1, se3_2)


def test_inverse():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3 = _create_random_se3(batch_size, rng)
        check_inverse(se3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3 = _create_random_se3(batch_size, rng)
        tangent = torch.randn(batch_size, 6).double()
        check_adjoint(se3, tangent)


def test_transform_from_and_to():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in [1, 20, 100]:
            se3 = _create_random_se3(batch_size, rng)
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
            se3 = _create_random_se3(batch_size, rng)
            point = th.Point3(data=torch.randn(batch_size, 3).double())

            # Test SE2.transform_to
            def transform_to_sum(g, p):
                return th.SE3(data=g).transform_to(p).data.sum(dim=0)

            jac = torch.autograd.functional.jacobian(
                transform_to_sum, (se3.data, point.data)
            )

            actual = [
                se3.project(jac[0]).transpose(0, 1),
                point.project(jac[1]).transpose(0, 1),
            ]
            expected = []
            _ = se3.transform_to(point, expected)
            assert torch.allclose(actual[0], expected[0])
            assert torch.allclose(actual[1], expected[1])

            # Test SE2.transform_from
            def transform_from_sum(g, p):
                return th.SE3(data=g).transform_from(p).data.sum(dim=0)

            jac = torch.autograd.functional.jacobian(
                transform_from_sum, (se3.data, point.data)
            )

            actual = [
                se3.project(jac[0]).transpose(0, 1),
                point.project(jac[1]).transpose(0, 1),
            ]
            expected = []
            _ = se3.transform_from(point, expected)
            assert torch.allclose(actual[0], expected[0])
            assert torch.allclose(actual[1], expected[1])
