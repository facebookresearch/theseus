# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import theseus as th
from tests.core.common import (
    BATCH_SIZES_TO_TEST,
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
)
from theseus.utils import numeric_jacobian


def evaluate_numerical_jacobian_between(Group, tol):
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in BATCH_SIZES_TO_TEST:
        v0 = Group.rand(batch_size, dtype=torch.float64, generator=rng)
        v1 = Group.rand(batch_size, dtype=torch.float64, generator=rng)
        measurement = Group.rand(batch_size, dtype=torch.float64, generator=rng)
        cost_function = th.Between(v0, v1, measurement, cost_weight)

        def new_error_fn(groups):
            new_cost_function = th.Between(
                groups[0], groups[1], measurement, cost_weight
            )
            return th.Vector(tensor=new_cost_function.error())

        expected_jacs = numeric_jacobian(new_error_fn, [v0, v1])
        jacobians, error_jac = cost_function.jacobians()
        error = cost_function.error()
        assert torch.allclose(error_jac, error)
        assert torch.allclose(jacobians[0], expected_jacs[0], atol=tol)
        assert torch.allclose(jacobians[1], expected_jacs[1], atol=tol)


def test_copy_between():
    v0 = th.SE2()
    v1 = th.SE2()
    meas = th.SE2()
    cost_function = th.Between(v0, v1, meas, th.ScaleCostWeight(1.0), name="name")
    cost_function2 = cost_function.copy(new_name="new_name")
    check_another_theseus_function_is_copy(
        cost_function, cost_function2, new_name="new_name"
    )
    check_another_theseus_tensor_is_copy(cost_function2.v0, v0)
    check_another_theseus_tensor_is_copy(cost_function2.v1, v1)
    check_another_theseus_tensor_is_copy(cost_function2.measurement, meas)
    check_another_theseus_function_is_copy(
        cost_function.weight,
        cost_function2.weight,
        new_name=f"{cost_function.weight.name}_copy",
    )
    assert cost_function2.name == "new_name"


def test_jacobian_between():
    evaluate_numerical_jacobian_between(th.SO2, 1e-8)
    evaluate_numerical_jacobian_between(th.SO3, 1e-6)
    evaluate_numerical_jacobian_between(th.SE2, 1e-8)
    evaluate_numerical_jacobian_between(th.SE3, 1e-6)


def test_error_between_point2():
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in BATCH_SIZES_TO_TEST:
        p1 = th.Point2(torch.randn(batch_size, 2, generator=rng))
        p2 = th.Point2(torch.randn(batch_size, 2, generator=rng))
        measurement = th.Point2(torch.randn(batch_size, 2, generator=rng))
        cost_function = th.Between(p1, p2, measurement, cost_weight)
        expected_error = (p2 - p1) - measurement
        error = cost_function.error()
        assert torch.allclose(expected_error.tensor, error)


def test_error_between_so2():
    so2_data = torch.DoubleTensor(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    ).unsqueeze(1)

    num_val = len(so2_data)
    between_errors = np.load("tests/embodied/measurements/between_errors_so2.npy")
    z = 0
    # between_errors was generated using all 3-combinations of the 5 SO2 above
    for i in range(num_val):
        for j in range(num_val):
            for k in range(num_val):
                meas = th.SO2(theta=so2_data[i, :1])
                so2_1 = th.SO2(theta=so2_data[j, :1])
                so2_2 = th.SO2(theta=so2_data[k, :1])
                dist_cf = th.Between(so2_1, so2_2, meas, th.ScaleCostWeight(1.0))
                assert np.allclose(dist_cf.error().item(), between_errors[z])
                z += 1


def test_error_between_se2():
    se2_data = torch.DoubleTensor(
        [
            [-1.1, 0.0, -np.pi],
            [0.0, 1.1, -np.pi / 2],
            [1.1, -1.1, 0.0],
            [0.0, 0.0, np.pi / 2],
            [-1.1, 1.1, np.pi],
        ]
    )
    num_val = len(se2_data)
    between_errors = np.load("tests/embodied/measurements/between_errors_se2.npy")
    z = 0
    # between_errors was generated using all 3-combinations of the 5 SE2 above
    for i in range(num_val):
        for j in range(num_val):
            for k in range(num_val):
                meas = th.SE2(x_y_theta=se2_data[i, :].unsqueeze(0))
                se2_1 = th.SE2(x_y_theta=se2_data[j, :].unsqueeze(0))
                se2_2 = th.SE2(x_y_theta=se2_data[k, :].unsqueeze(0))
                dist_cf = th.Between(se2_1, se2_2, meas, th.ScaleCostWeight(1.0))
                error = dist_cf.error()
                assert np.allclose(error.squeeze().numpy(), between_errors[z])
                z += 1


def test_jacobian_between_se3():
    for batch_size in BATCH_SIZES_TO_TEST:
        aux_id = torch.arange(batch_size)
        se3_1 = th.SE3.rand(batch_size, dtype=torch.float64)
        se3_2 = th.SE3.rand(batch_size, dtype=torch.float64)
        measurement = th.SE3.rand(batch_size, dtype=torch.float64)
        dist_cf = th.Between(se3_1, se3_2, measurement, th.ScaleCostWeight(1.0))

        actual = dist_cf.jacobians()[0]

        def test_fun(g1, g2):
            dist_cf.v0.update(g1)
            dist_cf.v1.update(g2)
            return dist_cf.error()

        jac_raw = torch.autograd.functional.jacobian(
            test_fun, (se3_1.tensor, se3_2.tensor)
        )
        expected = [
            se3_1.project(jac_raw[0][aux_id, :, aux_id], is_sparse=True),
            se3_2.project(jac_raw[1][aux_id, :, aux_id], is_sparse=True),
        ]

        assert torch.allclose(actual[0], expected[0])
        assert torch.allclose(actual[1], expected[1])
