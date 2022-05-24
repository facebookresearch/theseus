import copy

import pytest  # noqa: F401
import torch

import theseus as th

from .common import (
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
    create_mock_cost_functions,
)


def test_copy_scale_cost_weight():
    scale = th.Variable(torch.tensor(1.0))
    p1 = th.ScaleCostWeight(scale, name="scale_cost_weight")
    for the_copy in [p1.copy(), copy.deepcopy(p1)]:
        check_another_theseus_function_is_copy(p1, the_copy, new_name=f"{p1.name}_copy")
        check_another_theseus_tensor_is_copy(scale, the_copy.scale)


def test_copy_diagonal_cost_weight():
    diagonal = th.Variable(torch.ones(3))
    p1 = th.DiagonalCostWeight(diagonal, name="diagonal_cost_weight")
    for the_copy in [p1.copy(), copy.deepcopy(p1)]:
        check_another_theseus_function_is_copy(p1, the_copy, new_name=f"{p1.name}_copy")
        check_another_theseus_tensor_is_copy(diagonal, the_copy.diagonal)


def test_weight_error_scale_and_diagonal_cost_weight():
    def _check(f1_, f2_, scale_):
        j1, e1 = f1_.weighted_jacobians_error()
        j2, e2 = f2_.weighted_jacobians_error()
        assert torch.allclose(e1 * scale_, e2)
        for i in range(len(j1)):
            assert torch.allclose(j1[i] * scale_, j2[i])

    for _ in range(10):
        scale = torch.rand(1)
        cost_functions, *_ = create_mock_cost_functions(
            data=torch.ones(1, 10),
            cost_weight=th.ScaleCostWeight(th.Variable(torch.tensor(1.0))),
        )
        cost_functions_x_scale, *_ = create_mock_cost_functions(
            data=torch.ones(1, 10),
            cost_weight=th.ScaleCostWeight(th.Variable(scale.squeeze())),
        )
        for f1, f2 in zip(cost_functions, cost_functions_x_scale):
            _check(f1, f2, scale)

        diagonal = torch.ones(2)
        cost_functions, *_ = create_mock_cost_functions(
            data=torch.ones(1, 10),
            cost_weight=th.DiagonalCostWeight(th.Variable(diagonal)),
        )
        cost_functions_x_scale, *_ = create_mock_cost_functions(
            data=torch.ones(1, 10),
            cost_weight=th.DiagonalCostWeight(th.Variable(diagonal * scale)),
        )
        for f1, f2 in zip(cost_functions, cost_functions_x_scale):
            _check(f1, f2, scale)
