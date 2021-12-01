# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa
import torch

import theseus as th
from theseus.core.tests.common import (
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
    check_another_torch_tensor_is_copy,
)
from theseus.utils import numeric_jacobian


def test_collision2d_error_shapes():
    generator = torch.Generator()
    generator.manual_seed(0)
    cost_weight = th.ScaleCostWeight(1.0)
    for batch_size in [1, 10, 100]:
        for field_widht in [1, 10, 100]:
            for field_height in [1, 10, 100]:
                pose = th.Point2(data=torch.randn(batch_size, 2).double())
                origin = torch.randn(batch_size, 2)
                sdf_data = torch.randn(batch_size, field_widht, field_height)
                cell_size = torch.randn(batch_size, 1)
                cost_function = th.eb.Collision2D(
                    pose,
                    cost_weight,
                    th.Variable(origin),
                    th.Variable(sdf_data),
                    th.Variable(cell_size),
                    th.Variable(torch.ones(1)),
                    name="cost_function",
                )
                error = cost_function.error()
                jacobians, jac_error = cost_function.jacobians()
                assert error.shape == jac_error.shape
                assert error.shape == (batch_size, 1)
                assert jacobians[0].shape == (batch_size, 1, 2)


def test_collision2d_copy():
    batch_size = 20
    cost_weight = th.ScaleCostWeight(1.0)
    pose = th.Point2(data=torch.randn(batch_size, 2).double())
    origin = torch.ones(batch_size, 2)
    sdf_data = torch.ones(batch_size, 1, 1)
    cell_size = torch.ones(batch_size, 1)
    cost_function = th.eb.Collision2D(
        pose,
        cost_weight,
        th.Variable(origin),
        th.Variable(sdf_data),
        th.Variable(cell_size),
        th.Variable(torch.ones(1)),
        name="name",
    )
    cost_function2 = cost_function.copy(new_name="new_name")
    assert cost_function is not cost_function2
    check_another_theseus_tensor_is_copy(cost_function2.pose, pose)
    for attr_name in ["origin", "sdf_data", "cell_size"]:
        check_another_theseus_tensor_is_copy(
            getattr(cost_function.sdf, attr_name),
            getattr(cost_function2.sdf, attr_name),
        )
    check_another_theseus_function_is_copy(
        cost_function.weight,
        cost_function2.weight,
        new_name=f"{cost_function.weight.name}_copy",
    )
    check_another_torch_tensor_is_copy(
        cost_function.cost_eps.data, cost_function2.cost_eps.data
    )
    assert cost_function2.name == "new_name"


def test_collision2d_jacobians():
    for _ in range(10):
        for batch_size in [1, 10, 100, 1000]:
            cost_weight = th.ScaleCostWeight(torch.ones(1).squeeze().double())
            pose = th.Point2(data=torch.randn(batch_size, 2).double())
            origin = th.Variable(torch.ones(batch_size, 2).double())
            sdf_data = th.Variable(torch.randn(batch_size, 10, 10).double())
            cell_size = th.Variable(torch.rand(batch_size, 1).double())
            cost_eps = th.Variable(torch.rand(1).double())
            cost_function = th.eb.Collision2D(
                pose, cost_weight, origin, sdf_data, cell_size, cost_eps
            )

            def new_error_fn(vars):
                new_cost_function = th.eb.Collision2D(
                    vars[0], cost_weight, origin, sdf_data, cell_size, cost_eps
                )
                return th.Vector(data=new_cost_function.error())

            expected_jacs = numeric_jacobian(
                new_error_fn,
                [pose],
                function_dim=1,
                delta_mag=1e-6,
            )
            jacobians, error_jac = cost_function.jacobians()
            error = cost_function.error()
            assert torch.allclose(error_jac, error)
            assert torch.allclose(jacobians[0], expected_jacs[0], atol=1e-6)
