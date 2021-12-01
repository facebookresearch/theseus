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


def test_copy_between():
    v0 = th.SE2()
    v1 = th.SE2()
    meas = th.SE2()
    cost_function = th.eb.Between(v0, v1, th.ScaleCostWeight(1.0), meas, name="name")
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
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in [1, 10, 100]:
        v0 = create_random_se2(batch_size, rng)
        v1 = create_random_se2(batch_size, rng)
        measurement = create_random_se2(batch_size, rng)
        cost_function = th.eb.Between(v0, v1, cost_weight, measurement)

        def new_error_fn(groups):
            new_cost_function = th.eb.Between(
                groups[0], groups[1], cost_weight, measurement
            )
            return new_cost_function.measurement.retract(new_cost_function.error())

        expected_jacs = numeric_jacobian(new_error_fn, [v0, v1])
        jacobians, error_jac = cost_function.jacobians()
        error = cost_function.error()
        assert torch.allclose(error_jac, error)
        assert torch.allclose(jacobians[0], expected_jacs[0], atol=1e-8)
        assert torch.allclose(jacobians[1], expected_jacs[1], atol=1e-8)


def test_error_between_point2():
    rng = torch.Generator()
    rng.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1)
    for batch_size in [1, 10, 100]:
        p1 = th.Point2(torch.randn(batch_size, 2, generator=rng))
        p2 = th.Point2(torch.randn(batch_size, 2, generator=rng))
        measurement = th.Point2(torch.randn(batch_size, 2, generator=rng))
        cost_function = th.eb.Between(p1, p2, cost_weight, measurement)
        expected_error = (p2 - p1) - measurement
        error = cost_function.error()
        assert torch.allclose(expected_error.data, error.data)


def test_error_between_so2():
    so2_data = torch.DoubleTensor(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    ).unsqueeze(1)

    num_val = len(so2_data)
    between_errors = np.load(
        "theseus/embodied/measurements/tests/between_errors_so2.npy"
    )
    z = 0
    # between_errors was generated using all 3-combinations of the 5 SO2 above
    for i in range(num_val):
        for j in range(num_val):
            for k in range(num_val):
                meas = th.SO2(theta=so2_data[i, :1])
                so2_1 = th.SO2(theta=so2_data[j, :1])
                so2_2 = th.SO2(theta=so2_data[k, :1])
                dist_cf = th.eb.Between(
                    so2_1, so2_2, th.ScaleCostWeight(1.0), measurement=meas
                )
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
    between_errors = np.load(
        "theseus/embodied/measurements/tests/between_errors_se2.npy"
    )
    z = 0
    # between_errors was generated using all 3-combinations of the 5 SE2 above
    for i in range(num_val):
        for j in range(num_val):
            for k in range(num_val):
                meas = th.SE2(x_y_theta=se2_data[i, :].unsqueeze(0))
                se2_1 = th.SE2(x_y_theta=se2_data[j, :].unsqueeze(0))
                se2_2 = th.SE2(x_y_theta=se2_data[k, :].unsqueeze(0))
                dist_cf = th.eb.Between(
                    se2_1, se2_2, th.ScaleCostWeight(1.0), measurement=meas
                )
                error = dist_cf.error()
                assert np.allclose(error.squeeze().numpy(), between_errors[z])
                z += 1
