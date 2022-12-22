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


def evaluate_numerical_jacobian_local_cost_fn(Group, tol):
    rng = torch.Generator()
    rng.manual_seed(1)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in BATCH_SIZES_TO_TEST:
        v0 = Group.rand(batch_size, dtype=torch.float64, generator=rng)
        target = Group.rand(batch_size, dtype=torch.float64, generator=rng)
        cost_function = th.Difference(v0, target, cost_weight)

        def new_error_fn(groups):
            new_cost_function = th.Difference(groups[0], target, cost_weight)
            return th.Vector(tensor=new_cost_function.error())

        expected_jacs = numeric_jacobian(new_error_fn, [v0])
        jacobians, error_jac = cost_function.jacobians()
        error = cost_function.error()
        assert torch.allclose(error_jac, error)
        assert torch.allclose(jacobians[0], expected_jacs[0], atol=tol)


def test_copy_local_cost_fn():
    v0 = th.Vector(tensor=torch.zeros(1, 1))
    target = th.Vector(tensor=torch.ones(1, 1))
    cost_function = th.Difference(v0, target, th.ScaleCostWeight(1.0), name="name")
    cost_function2 = cost_function.copy(new_name="new_name")
    check_another_theseus_function_is_copy(
        cost_function, cost_function2, new_name="new_name"
    )
    check_another_theseus_tensor_is_copy(cost_function2.var, v0)
    check_another_theseus_tensor_is_copy(cost_function2.target, target)
    check_another_theseus_function_is_copy(
        cost_function.weight,
        cost_function2.weight,
        new_name=f"{cost_function.weight.name}_copy",
    )
    assert cost_function2.name == "new_name"


def test_jacobian_local_cost_fn():
    evaluate_numerical_jacobian_local_cost_fn(th.SO2, 1e-6)
    evaluate_numerical_jacobian_local_cost_fn(th.SE2, 1e-8)
    evaluate_numerical_jacobian_local_cost_fn(th.SO3, 1e-6)
    evaluate_numerical_jacobian_local_cost_fn(th.SE3, 1e-6)


def test_error_local_cost_fn_point2():
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in BATCH_SIZES_TO_TEST:
        p0 = th.Point2(torch.randn(batch_size, 2, generator=rng))
        target = th.Point2(torch.randn(batch_size, 2, generator=rng))
        cost_function = th.Difference(p0, target, cost_weight)
        expected_error = p0 - target
        error = cost_function.error()
        assert torch.allclose(expected_error.tensor, error)


def test_error_local_cost_fn_so2():
    so2_data = torch.DoubleTensor(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    ).unsqueeze(1)

    num_val = len(so2_data)
    sq_dist_errors = np.load("tests/embodied/misc/sq_dist_errors_so2.npy")
    k = 0
    # sq_dist_errors was generated using all 2-combinations of the 5 SO2 above
    for i in range(num_val):
        for j in range(num_val):
            meas = th.SO2(theta=so2_data[i, :1])
            so2 = th.SO2(theta=so2_data[j, :1])
            dist_cf = th.Difference(so2, meas, th.ScaleCostWeight(1.0))
            assert np.allclose(dist_cf.error().squeeze().item(), sq_dist_errors[k])
            k += 1


def test_error_local_cost_fn_se2():
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
    sq_dist_errors = np.load("tests/embodied/misc/sq_dist_errors_se2.npy")
    k = 0
    # sq_dist_errors was generated using all 2-combinations of the 5 SE2 above
    for i in range(num_val):
        for j in range(num_val):
            meas = th.SE2(x_y_theta=se2_data[i, :].unsqueeze(0))
            se2 = th.SE2(x_y_theta=se2_data[j, :].unsqueeze(0))
            dist_cf = th.Difference(se2, meas, th.ScaleCostWeight(1.0))
            error = dist_cf.error()
            assert np.allclose(error.squeeze().numpy(), sq_dist_errors[k])
            k += 1
