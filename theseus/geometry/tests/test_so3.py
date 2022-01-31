# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS

from .common import check_adjoint, check_compose, check_exp_map


def _create_random_so3(batch_size, rng):
    q = torch.rand(batch_size, 4, generator=rng).double() - 0.5
    qnorm = torch.linalg.norm(q, dim=1, keepdim=True)
    q = q / qnorm
    return th.SO3(quaternion=q)


def check_SO3_log_map(tangent_vector):
    error = (tangent_vector - th.SO3.exp_map(tangent_vector).log_map()).norm(dim=1)
    error = torch.minimum(error, (error - 2 * np.pi).abs())
    assert torch.allclose(error, torch.zeros_like(error), atol=EPS)


def check_SO3_to_quaternion(so3: th.SO3):
    quaternions = so3.to_quaternion()
    assert torch.allclose(
        th.SO3(quaternion=quaternions).to_matrix(), so3.to_matrix(), atol=1e-8
    )


def test_exp_map():
    for batch_size in [1, 20, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-5
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 3e-3
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        check_exp_map(tangent_vector, th.SO3)


def test_log_map():
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        check_SO3_log_map(tangent_vector)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-6
        check_SO3_log_map(tangent_vector)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-3
        check_SO3_log_map(tangent_vector)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        check_SO3_log_map(tangent_vector)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-3
        check_SO3_log_map(tangent_vector)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        check_SO3_log_map(tangent_vector)


def test_quaternion():
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-6
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-3
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-3
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        so3 = th.SO3.exp_map(tangent_vector)
        check_SO3_to_quaternion(so3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3 = _create_random_so3(batch_size, rng)
        tangent = torch.randn(batch_size, 3).double()
        check_adjoint(so3, tangent)


def test_compose():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3_1 = _create_random_so3(batch_size, rng)
        so3_2 = _create_random_so3(batch_size, rng)
        check_compose(so3_1, so3_2)


# def test_rotate_and_unrotate():
#     rng = torch.Generator()
#     rng.manual_seed(0)
#     for _ in range(10):  # repeat a few times
#         for batch_size in [1, 20, 100]:
#             so3 = _create_random_so3(batch_size, rng)
#             # Tests that rotate works from tensor. unrotate() would work similarly), but
#             # it's also tested indirectly by test_transform_to() for SE2
#             point_tensor = torch.randn(batch_size, 3).double()
#             jacobians_rotate = []
#             rotated_point = so3.rotate(point_tensor, jacobians=jacobians_rotate)
#             expected_rotated_data = so3.to_matrix() @ point_tensor.unsqueeze(2)
#             jacobians_unrotate = []
#             unrotated_point = so3.unrotate(rotated_point, jacobians_unrotate)

#             # Check the operation result
#             assert torch.allclose(
#                 expected_rotated_data.squeeze(2), rotated_point.data, atol=EPS
#             )
#             assert torch.allclose(point_tensor, unrotated_point.data, atol=EPS)
