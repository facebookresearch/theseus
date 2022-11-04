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
    check_projection_for_compose,
    check_projection_for_exp_map,
    check_projection_for_inverse,
    check_projection_for_log_map,
    check_projection_for_rotate_and_transform,
    check_so3_se3_normalize,
)


def check_SE3_log_map(tangent_vector, atol=TEST_EPS, enable_functorch=False):
    with set_lie_group_check_enabled(not enable_functorch, silent=True):
        g = th.SE3.exp_map(tangent_vector)
        assert torch.allclose(th.SE3.exp_map(g.log_map()).tensor, g.tensor, atol=atol)


def check_SE3_to_x_y_z_quaternion(se3: th.SE3, atol=1e-10, enable_functorch=False):
    with set_lie_group_check_enabled(not enable_functorch, silent=True):
        x_y_z_quaternion = se3.to_x_y_z_quaternion()
        assert torch.allclose(
            th.SE3(x_y_z_quaternion=x_y_z_quaternion).to_matrix(),
            se3.to_matrix(),
            atol=atol,
        )


def _create_tangent_vector(batch_size, ang_factor, rng, dtype):
    tangent_vector_ang = torch.rand(batch_size, 3, generator=rng, dtype=dtype) - 0.5
    tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
    tangent_vector_ang *= ang_factor
    tangent_vector_lin = torch.randn(batch_size, 3, generator=rng, dtype=dtype)
    tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)
    return tangent_vector


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-11]
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
    check_exp_map(tangent_vector, th.SE3, atol=ATOL, enable_functorch=enable_functorch)
    check_projection_for_exp_map(
        tangent_vector, th.SE3, atol=ATOL, enable_functorch=enable_functorch
    )


# This test checks that cross products are done correctly
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-11]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_batch_size_3_exp_map(dtype, ang_factor, enable_functorch):
    with set_lie_group_check_enabled(not enable_functorch, silent=True):
        rng = torch.Generator()
        rng.manual_seed(0)
        ATOL = 1e-4 if dtype == torch.float32 else 1e-6

        if ang_factor is None:
            ang_factor = (
                torch.rand(6, 1, generator=rng, dtype=dtype) * 2 * np.pi - np.pi
            )
        tangent_vector = _create_tangent_vector(6, ang_factor, rng, dtype)

        jac, jac1, jac2 = [], [], []
        g = th.SE3.exp_map(tangent_vector, jac)
        g1 = th.SE3.exp_map(tangent_vector[:3], jac1)
        g2 = th.SE3.exp_map(tangent_vector[3:], jac2)

        torch.allclose(g.tensor[:3], g1.tensor, atol=1e-6)
        torch.allclose(g.tensor[3:], g2.tensor, atol=1e-6)
        torch.allclose(jac[0][:3], jac1[0], atol=ATOL)
        torch.allclose(jac[0][3:], jac2[0], atol=ATOL)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-7, np.pi - 1e-10]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_log_map(batch_size, dtype, ang_factor, enable_functorch):
    if dtype == torch.float32 and ang_factor == np.pi - 1e-10:
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
    check_SE3_log_map(tangent_vector, atol=ATOL, enable_functorch=enable_functorch)
    check_projection_for_log_map(
        tangent_vector, th.SE3, atol=PROJECTION_ATOL, enable_functorch=enable_functorch
    )


# This test checks that cross products are done correctly
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-11]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_batch_size_3_log_map(dtype, ang_factor, enable_functorch):
    with set_lie_group_check_enabled(not enable_functorch, silent=True):
        rng = torch.Generator()
        rng.manual_seed(0)
        ATOL = 1e-3 if dtype == torch.float32 else 1e-6

        if ang_factor is None:
            ang_factor = (
                torch.rand(6, 1, generator=rng, dtype=dtype) * 2 * np.pi - np.pi
            )
        tangent_vector = _create_tangent_vector(6, ang_factor, rng, dtype)

        g = th.SE3.exp_map(tangent_vector)
        g1 = th.SE3(tensor=g.tensor[:3])
        g2 = th.SE3(tensor=g.tensor[3:])

        jac, jac1, jac2 = [], [], []
        d = g.log_map(jac)
        d1 = g1.log_map(jac1)
        d2 = g2.log_map(jac2)

        torch.allclose(d[:3], d1, atol=ATOL)
        torch.allclose(d[3:], d2, atol=ATOL)
        torch.allclose(jac[0][:3], jac1[0], atol=ATOL)
        torch.allclose(jac[0][3:], jac2[0], atol=ATOL)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_compose(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se3_1 = th.SE3.rand(batch_size, generator=rng, dtype=dtype)
        se3_2 = th.SE3.rand(batch_size, generator=rng, dtype=dtype)
        check_compose(se3_1, se3_2, enable_functorch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_inverse(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        se3 = th.SE3.rand(batch_size, generator=rng, dtype=dtype)
        check_inverse(se3, enable_functorch)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "ang_factor", [None, 1e-5, 3e-3, 2 * np.pi - 1e-11, np.pi - 1e-11]
)
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_x_y_z_quaternion(batch_size, dtype, ang_factor, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 1e-3 if dtype == torch.float32 else 1e-8

    if ang_factor is None:
        ang_factor = (
            torch.rand(batch_size, 1, generator=rng, dtype=dtype) * 2 * np.pi - np.pi
        )

    tangent_vector = _create_tangent_vector(batch_size, ang_factor, rng, dtype)
    se3 = th.SE3.exp_map(tangent_vector)
    check_SE3_to_x_y_z_quaternion(se3, atol=ATOL, enable_functorch=enable_functorch)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_adjoint(batch_size, dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    se3 = th.SE3.rand(batch_size, generator=rng, dtype=dtype)
    tangent = torch.randn(batch_size, 6, generator=rng, dtype=dtype)
    check_adjoint(se3, tangent, enable_functorch)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transform_from_and_to(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size_group in [1, 20]:
            for batch_size_pnt in [1, 20]:
                if (
                    batch_size_group != 1
                    and batch_size_pnt != 1
                    and batch_size_pnt != batch_size_group
                ):
                    continue
                se3 = th.SE3.rand(batch_size_group, generator=rng, dtype=dtype)
                point_tensor = torch.randn(
                    batch_size_pnt, 3, generator=rng, dtype=dtype
                )
                point_tensor_ext = torch.cat(
                    (point_tensor, torch.ones(batch_size_pnt, 1, dtype=dtype)), dim=1
                )

                jacobians_to = []
                point_to = se3.transform_to(point_tensor, jacobians=jacobians_to)
                expected_to = (
                    se3.inverse().to_matrix().double()
                    @ point_tensor_ext.double().unsqueeze(2)
                )[:, :3].squeeze(2)
                jacobians_from = []
                point_from = se3.transform_from(point_to, jacobians_from)

                # Check the operation result
                assert torch.allclose(
                    expected_to, point_to.tensor.double(), atol=TEST_EPS
                )
                assert torch.allclose(point_tensor, point_from.tensor, atol=5e-7)

                # Check the jacobians
                se3_double = se3.copy()
                se3_double.to(torch.float64)
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].transform_to(groups[1]),
                    [se3_double, th.Point3(point_tensor.double())],
                    function_dim=3,
                )
                assert torch.allclose(
                    jacobians_to[0].double(), expected_jac[0], atol=5e-7
                )
                assert torch.allclose(
                    jacobians_to[1].double(), expected_jac[1], atol=TEST_EPS
                )
                expected_jac = numeric_jacobian(
                    lambda groups: groups[0].transform_from(groups[1]),
                    [se3_double, th.Vector(tensor=point_to.tensor.double())],
                    delta_mag=1e-5,
                    function_dim=3,
                )
                assert torch.allclose(
                    jacobians_from[0].double(), expected_jac[0], atol=TEST_EPS
                )
                assert torch.allclose(
                    jacobians_from[1].double(), expected_jac[1], atol=TEST_EPS
                )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("enable_functorch", [True, False])
def test_projection(dtype, enable_functorch):
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a few times
        for batch_size in BATCH_SIZES_TO_TEST:
            # Test SE3.transform_to
            check_projection_for_rotate_and_transform(
                th.SE3, th.Point3, th.SE3.transform_to, batch_size, rng, dtype=dtype
            )

            # Test SE3.transform_from
            check_projection_for_rotate_and_transform(
                th.SE3, th.Point3, th.SE3.transform_from, batch_size, rng, dtype=dtype
            )

            # Test SE3.compose
            check_projection_for_compose(
                th.SE3, batch_size, rng, dtype=dtype, enable_functorch=enable_functorch
            )

            # Test SE3.inverse
            check_projection_for_inverse(
                th.SE3, batch_size, rng, dtype=dtype, enable_functorch=enable_functorch
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_map(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    ATOL = 3e-5 if dtype == torch.float32 else 1e-7

    for batch_size in BATCH_SIZES_TO_TEST:
        group0 = th.SE3.rand(batch_size, dtype=dtype, generator=rng)
        group1 = th.SE3.rand(batch_size, dtype=dtype, generator=rng)
        check_jacobian_for_local(
            group0, group1, Group=th.SE3, is_projected=True, atol=ATOL
        )


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalization(batch_size, dtype):
    check_so3_se3_normalize(th.SE3, batch_size, dtype)
