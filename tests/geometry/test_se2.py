# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import TEST_EPS
from tests.core.common import check_copy_var
from theseus.utils import numeric_jacobian

from .common import (
    BATCH_SIZES_TO_TEST,
    check_adjoint,
    check_compose,
    check_exp_map,
    check_inverse,
    check_log_map,
    check_normalize,
    check_projection_for_compose,
    check_projection_for_exp_map,
    check_projection_for_inverse,
    check_projection_for_rotate_and_transform,
)


def create_random_se2(batch_size, rng, dtype=torch.float64):
    theta = torch.rand(batch_size, 1, generator=rng) * 2 * np.pi - np.pi
    u = torch.randn(batch_size, 2)
    tangent_vector = torch.cat([u, theta], dim=1)
    return th.SE2.exp_map(tangent_vector.to(dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_exp_map(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 2e-4 if dtype == torch.float32 else 1e-6

    for batch_size in BATCH_SIZES_TO_TEST:
        theta = torch.from_numpy(np.linspace(-np.pi, np.pi, batch_size))
        u = torch.randn(batch_size, 2, dtype=dtype, generator=rng)
        tangent_vector = torch.cat([u, theta.unsqueeze(1)], dim=1)
        check_exp_map(
            tangent_vector.to(dtype=dtype), th.SE2, enable_functorch=enable_functorch
        )
        check_projection_for_exp_map(
            tangent_vector.to(dtype=dtype),
            th.SE2,
            atol=ATOL,
            enable_functorch=enable_functorch,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_log_map(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        theta = torch.from_numpy(np.linspace(-np.pi, np.pi, batch_size))
        u = torch.randn(batch_size, 2, dtype=dtype, generator=rng)
        tangent_vector = torch.cat([u, theta.unsqueeze(1)], dim=1)
        check_log_map(tangent_vector, th.SE2, enable_functorch=enable_functorch)
        check_projection_for_exp_map(
            tangent_vector, th.SE2, enable_functorch=enable_functorch
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_compose(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se2_1 = th.SE2.rand(batch_size, generator=rng, dtype=dtype)
        se2_2 = th.SE2.rand(batch_size, generator=rng, dtype=dtype)
        check_compose(se2_1, se2_2, enable_functorch=enable_functorch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_inverse(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=dtype)
        check_inverse(se2, enable_functorch=enable_functorch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_adjoint(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=dtype)
        tangent = torch.randn(batch_size, 3, dtype=dtype)
        check_adjoint(se2, tangent, enable_functorch=enable_functorch)


def test_copy():
    rng = torch.Generator()
    se2 = th.SE2.rand(1, generator=rng, dtype=torch.float64)
    check_copy_var(se2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transform_from_and_to(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size_se2 in BATCH_SIZES_TO_TEST:
            for batch_size_pnt in BATCH_SIZES_TO_TEST:
                if (
                    batch_size_se2 != 1
                    and batch_size_pnt != 1
                    and batch_size_pnt != batch_size_se2
                ):
                    continue

                se2 = th.SE2.rand(batch_size_se2, generator=rng, dtype=dtype)
                point_tensor = torch.randn(batch_size_pnt, 2, dtype=dtype)
                point_tensor_ext = torch.cat(
                    [point_tensor, torch.ones(batch_size_pnt, 1, dtype=dtype)], dim=1
                )

                jacobians_to = []
                point_to = se2.transform_to(point_tensor, jacobians=jacobians_to)
                expected_to = (
                    se2.inverse().to_matrix().double()
                    @ point_tensor_ext.unsqueeze(2).double()
                )[:, :2]
                jacobians_from = []
                point_from = se2.transform_from(point_to, jacobians_from)

                # Check the operation result
                assert torch.allclose(
                    expected_to.squeeze(2), point_to.tensor.double(), atol=TEST_EPS
                )
                assert torch.allclose(
                    point_tensor.double(), point_from.tensor.double(), atol=TEST_EPS
                )

                # Check the jacobians
                se2_double = se2.copy()
                se2_double.to(torch.float64)
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].transform_to(groups[1]),
                    [se2_double, th.Point2(point_tensor.double())],
                    function_dim=2,
                )

                assert jacobians_to[0].shape == expected_jac[0].shape
                assert jacobians_to[1].shape == expected_jac[1].shape
                assert torch.allclose(
                    jacobians_to[0].double(), expected_jac[0], atol=TEST_EPS
                )
                assert torch.allclose(
                    jacobians_to[1].double(), expected_jac[1], atol=TEST_EPS
                )

                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].transform_from(groups[1]),
                    [se2_double, th.Point2(point_to.tensor.double())],
                    function_dim=2,
                )
                assert jacobians_from[0].shape == expected_jac[0].shape
                assert jacobians_from[1].shape == expected_jac[1].shape
                assert torch.allclose(
                    jacobians_from[0].double(), expected_jac[0], atol=TEST_EPS
                )
                assert torch.allclose(
                    jacobians_from[1].double(), expected_jac[1], atol=TEST_EPS
                )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_xy_jacobian(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=dtype)
        jacobian = []
        se2.xy(jacobians=jacobian)
        expected_jac = numeric_jacobian(
            lambda groups: groups[0].xy(), [se2], function_dim=2
        )
        torch.allclose(jacobian[0], expected_jac[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_theta_jacobian(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se2 = th.SE2.rand(batch_size, generator=rng, dtype=dtype)
        jacobian = []
        se2.theta(jacobians=jacobian)
        expected_jac = numeric_jacobian(
            lambda groups: th.Vector(tensor=groups[0].theta()), [se2], function_dim=1
        )
        torch.allclose(jacobian[0], expected_jac[0])


def test_projection():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in BATCH_SIZES_TO_TEST:
            # Test SE2.transform_to
            check_projection_for_rotate_and_transform(
                th.SE2, th.Point2, th.SE2.transform_to, batch_size, rng
            )

            # Test SE2.transform_from
            check_projection_for_rotate_and_transform(
                th.SE2, th.Point2, th.SE2.transform_from, batch_size, rng
            )

            # Test SE2.compose
            check_projection_for_compose(th.SE2, batch_size, rng)

            # Test SE2.inverse
            check_projection_for_inverse(th.SE2, batch_size, rng)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalization(batch_size, dtype):
    check_normalize(th.SE2, batch_size, dtype)
