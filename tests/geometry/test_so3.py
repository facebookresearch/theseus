# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import TEST_EPS
from theseus.geometry.lie_group_check import set_lie_group_check_enabled
from theseus.utils import numeric_jacobian

from .common import (
    BATCH_SIZES_TO_TEST,
    check_adjoint,
    check_compose,
    check_exp_map,
    check_inverse,
    check_jacobian_for_local,
    check_log_map,
    check_projection_for_compose,
    check_projection_for_exp_map,
    check_projection_for_inverse,
    check_projection_for_log_map,
    check_projection_for_rotate_and_transform,
    check_so3_se3_normalize,
)


def check_SO3_log_map(tangent_vector, atol=1e-7, enable_functorch=False):
    with set_lie_group_check_enabled(not enable_functorch, silent=True):
        error = (tangent_vector - th.SO3.exp_map(tangent_vector).log_map()).norm(dim=1)
        error = torch.minimum(error, (error - 2 * np.pi).abs())
        assert torch.allclose(error, torch.zeros_like(error), atol=atol)


def check_SO3_to_quaternion(so3: th.SO3, atol=1e-10, enable_functorch=False):
    with set_lie_group_check_enabled(not enable_functorch, silent=True):
        quaternions = so3.to_quaternion()
        assert torch.allclose(
            th.SO3(quaternion=quaternions).to_matrix(), so3.to_matrix(), atol=atol
        )


def _create_tangent_vector(batch_size, ang_factor, rng, dtype):
    tangent_vector = torch.rand(batch_size, 3, generator=rng, dtype=dtype) - 0.5
    tangent_vector /= tangent_vector.norm(dim=1, keepdim=True)
    tangent_vector *= ang_factor
    return tangent_vector


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-7, np.pi - 1e-11]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_exp_map(batch_size, dtype, ang_factor, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 2e-4 if dtype == torch.float32 else 1e-6

    if ang_factor is None:
        ang_factor = (
            torch.rand(batch_size, 1, generator=rng, dtype=dtype) * 2 * np.pi - np.pi
        )

    tangent_vector = _create_tangent_vector(batch_size, ang_factor, rng, dtype)
    check_exp_map(tangent_vector, th.SO3, atol=ATOL, enable_functorch=enable_functorch)
    check_projection_for_exp_map(
        tangent_vector, th.SO3, atol=ATOL, enable_functorch=enable_functorch
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-11]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_log_map(batch_size, dtype, ang_factor, enable_functorch):
    if dtype == torch.float32 and ang_factor == np.pi - 1e-11:
        return

    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 1e-3 if dtype == torch.float32 else 1e-8
    PROJECTION_ATOL = 1e-2 if dtype == torch.float32 else 1e-8

    if ang_factor is None:
        ang_factor = (
            torch.rand(batch_size, 1, generator=rng, dtype=dtype) * 2 * np.pi - np.pi
        )

    tangent_vector = _create_tangent_vector(batch_size, ang_factor, rng, dtype)
    check_SO3_log_map(tangent_vector, atol=ATOL, enable_functorch=enable_functorch)
    if tangent_vector.norm(dim=1).max() < 2 * np.pi - 1e-3:
        check_log_map(
            tangent_vector, th.SO3, atol=ATOL, enable_functorch=enable_functorch
        )
    check_projection_for_log_map(
        tangent_vector, th.SO3, atol=PROJECTION_ATOL, enable_functorch=enable_functorch
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_inverse(batch_size, dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)

    group = th.SO3.rand(batch_size, generator=rng, dtype=dtype)

    check_inverse(group, enable_functorch=enable_functorch)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-11]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_quaternion(batch_size, dtype, ang_factor, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 1e-3 if dtype == torch.float32 else 1e-8

    if ang_factor is None:
        ang_factor = (
            torch.rand(batch_size, 1, generator=rng, dtype=dtype) * 2 * np.pi - np.pi
        )

    tangent_vector = _create_tangent_vector(batch_size, ang_factor, rng, dtype)
    so3 = th.SO3.exp_map(tangent_vector)
    check_SO3_to_quaternion(so3, atol=ATOL, enable_functorch=enable_functorch)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_adjoint(batch_size, dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    so3 = th.SO3.rand(batch_size, generator=rng, dtype=dtype)
    tangent = torch.randn(batch_size, 3, dtype=dtype)
    check_adjoint(so3, tangent, enable_functorch)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compose(batch_size, dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    so3_1 = th.SO3.rand(batch_size, generator=rng, dtype=dtype)
    so3_2 = th.SO3.rand(batch_size, generator=rng, dtype=dtype)
    check_compose(so3_1, so3_2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotate_and_unrotate(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size_group in BATCH_SIZES_TO_TEST:
            for batch_size_pnt in BATCH_SIZES_TO_TEST:
                if (
                    batch_size_group != 1
                    and batch_size_pnt != 1
                    and batch_size_pnt != batch_size_group
                ):
                    continue

                so3 = th.SO3.rand(batch_size_group, generator=rng, dtype=dtype)
                point_tensor = torch.randn(
                    batch_size_pnt, 3, generator=rng, dtype=dtype
                )

                jacobians_rotate = []
                rotated_point = so3.rotate(point_tensor, jacobians=jacobians_rotate)
                expected_rotated_data = so3.to_matrix() @ point_tensor.unsqueeze(2)
                jacobians_unrotate = []
                unrotated_point = so3.unrotate(rotated_point, jacobians_unrotate)

                # Check the operation result
                assert torch.allclose(
                    expected_rotated_data.squeeze(2),
                    rotated_point.tensor,
                    atol=TEST_EPS,
                )
                assert torch.allclose(
                    point_tensor, unrotated_point.tensor, atol=TEST_EPS
                )

                # Check the jacobians
                # function_dim = 3 because rotate(so3, (x, y, z)) --> (x_new, y_new, z_new)
                so3_double = so3.copy()
                so3_double.to(torch.float64)
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].rotate(groups[1]),
                    [so3_double, th.Point3(point_tensor.double())],
                    function_dim=3,
                )
                assert torch.allclose(
                    jacobians_rotate[0].double(), expected_jac[0], atol=TEST_EPS
                )
                assert torch.allclose(
                    jacobians_rotate[1].double(), expected_jac[1], atol=TEST_EPS
                )
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].unrotate(groups[1]),
                    [so3_double, th.Vector(tensor=rotated_point.tensor.double())],
                    delta_mag=1e-5,
                    function_dim=3,
                )
                assert torch.allclose(
                    jacobians_unrotate[0].double(), expected_jac[0], atol=TEST_EPS
                )
                assert torch.allclose(
                    jacobians_unrotate[1].double(), expected_jac[1], atol=TEST_EPS
                )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_projection(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in BATCH_SIZES_TO_TEST:
            # Test SO3.rotate
            check_projection_for_rotate_and_transform(
                th.SO3, th.Point3, th.SO3.rotate, batch_size, rng, dtype=dtype
            )

            # Test SO3.unrotate
            check_projection_for_rotate_and_transform(
                th.SO3, th.Point3, th.SO3.unrotate, batch_size, rng, dtype=dtype
            )

            # Test SO3.compose
            check_projection_for_compose(
                th.SO3, batch_size, rng, dtype=dtype, enable_functorch=enable_functorch
            )

            # Test SO3.inverse
            check_projection_for_inverse(
                th.SO3, batch_size, rng, dtype=dtype, enable_functorch=enable_functorch
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_map(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 3e-5 if dtype == torch.float32 else 1e-7

    for batch_size in BATCH_SIZES_TO_TEST:
        group0 = th.SO3.rand(batch_size, dtype=dtype)
        group1 = th.SO3.rand(batch_size, dtype=dtype)

        check_jacobian_for_local(
            group0, group1, Group=th.SO3, is_projected=True, atol=ATOL
        )


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalization(batch_size, dtype):
    check_so3_se3_normalize(th.SO3, batch_size, dtype)
