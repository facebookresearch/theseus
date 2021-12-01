# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import theseus as th
from theseus.core.tests.common import (
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
)
from theseus.geometry.tests.test_se2 import create_random_se2
from theseus.utils import numeric_jacobian


def test_copy_variable_difference():
    v0 = th.Vector(data=torch.zeros(1, 1))
    target = th.Vector(data=torch.ones(1, 1))
    cost_function = th.eb.VariableDifference(
        v0, th.ScaleCostWeight(1.0), target, name="name"
    )
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


def test_jacobian_variable_difference():
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in [1, 10, 100]:
        v0 = create_random_se2(batch_size, rng)
        target = create_random_se2(batch_size, rng)
        cost_function = th.eb.VariableDifference(v0, cost_weight, target)

        def new_error_fn(groups):
            new_cost_function = th.eb.VariableDifference(groups[0], cost_weight, target)
            return new_cost_function.target.retract(new_cost_function.error())

        expected_jacs = numeric_jacobian(new_error_fn, [v0])
        jacobians, error_jac = cost_function.jacobians()
        error = cost_function.error()
        assert torch.allclose(error_jac, error)
        assert torch.allclose(jacobians[0], expected_jacs[0], atol=1e-8)


def test_error_variable_difference_point2():
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in [1, 10, 100]:
        p0 = th.Point2(torch.randn(batch_size, 2, generator=rng))
        target = th.Point2(torch.randn(batch_size, 2, generator=rng))
        cost_function = th.eb.VariableDifference(p0, cost_weight, target)
        expected_error = p0 - target
        error = cost_function.error()
        assert torch.allclose(expected_error.data, error.data)


def test_error_variable_difference_so2():
    so2_data = torch.DoubleTensor(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    ).unsqueeze(1)

    num_val = len(so2_data)
    sq_dist_errors = np.load("theseus/embodied/misc/tests/sq_dist_errors_so2.npy")
    k = 0
    # sq_dist_errors was generated using all 2-combinations of the 5 SO2 above
    for i in range(num_val):
        for j in range(num_val):
            meas = th.SO2(theta=so2_data[i, :1])
            so2 = th.SO2(theta=so2_data[j, :1])
            dist_cf = th.eb.VariableDifference(
                so2, th.ScaleCostWeight(1.0), target=meas
            )
            assert np.allclose(dist_cf.error().squeeze().item(), sq_dist_errors[k])
            k += 1


def test_error_variable_difference_se2():
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
    sq_dist_errors = np.load("theseus/embodied/misc/tests/sq_dist_errors_se2.npy")
    k = 0
    # sq_dist_errors was generated using all 2-combinations of the 5 SE2 above
    for i in range(num_val):
        for j in range(num_val):
            meas = th.SE2(x_y_theta=se2_data[i, :].unsqueeze(0))
            se2 = th.SE2(x_y_theta=se2_data[j, :].unsqueeze(0))
            dist_cf = th.eb.VariableDifference(
                se2, th.ScaleCostWeight(1.0), target=meas
            )
            error = dist_cf.error()
            assert np.allclose(error.squeeze().numpy(), sq_dist_errors[k])
            k += 1
