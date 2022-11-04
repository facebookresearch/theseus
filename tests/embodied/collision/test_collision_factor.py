# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa
import torch

import theseus as th
from tests.core.common import (
    BATCH_SIZES_TO_TEST,
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
    check_another_torch_tensor_is_copy,
)
from theseus.utils import numeric_jacobian

from .utils import random_origin, random_scalar, random_sdf_data


def test_collision2d_error_shapes():
    cost_weight = th.ScaleCostWeight(1.0)
    for batch_size in BATCH_SIZES_TO_TEST:
        for field_widht in BATCH_SIZES_TO_TEST:
            for field_height in BATCH_SIZES_TO_TEST:
                pose = th.Point2(tensor=torch.randn(batch_size, 2).double())
                origin = random_origin(batch_size)
                sdf_data = random_sdf_data(batch_size, field_widht, field_height)
                cell_size = random_scalar(batch_size)
                cost_function = th.eb.Collision2D(
                    pose,
                    origin,
                    sdf_data,
                    cell_size,
                    th.Variable(torch.ones(1)),
                    cost_weight,
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
    pose = th.Point2(tensor=torch.randn(batch_size, 2).double())
    origin = random_origin(batch_size)
    sdf_data = random_sdf_data(batch_size, 1, 1)
    cell_size = random_scalar(batch_size)
    cost_function = th.eb.Collision2D(
        pose,
        origin,
        sdf_data,
        cell_size,
        1.0,
        cost_weight,
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
        cost_function.cost_eps.tensor, cost_function2.cost_eps.tensor
    )
    assert cost_function2.name == "new_name"


@pytest.mark.parametrize("pose_cls", [th.Point2, th.SE2])
def test_collision2d_jacobians(pose_cls):
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):
        for batch_size in BATCH_SIZES_TO_TEST:
            cost_weight = th.ScaleCostWeight(torch.ones(1).squeeze().double())
            pose = pose_cls.randn(batch_size, generator=rng, dtype=torch.float64)
            origin = th.Point2(torch.ones(batch_size, 2).double())
            sdf_data = th.Variable(
                torch.randn(batch_size, 10, 10, generator=rng).double()
            )
            cell_size = th.Variable(torch.rand(batch_size, 1, generator=rng).double())
            cost_eps = th.Variable(torch.rand(1, generator=rng).double())
            cost_function = th.eb.Collision2D(
                pose, origin, sdf_data, cell_size, cost_eps, cost_weight
            )

            def new_error_fn(vars):
                new_cost_function = th.eb.Collision2D(
                    vars[0], origin, sdf_data, cell_size, cost_eps, cost_weight
                )
                return th.Vector(tensor=new_cost_function.error())

            expected_jacs = numeric_jacobian(
                new_error_fn,
                [pose],
                function_dim=1,
                delta_mag=1e-6,
            )
            jacobians, error_jac = cost_function.jacobians()
            error = cost_function.error()
            assert torch.allclose(error_jac, error)
            assert torch.allclose(jacobians[0], expected_jacs[0], atol=1e-5)
