# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.core import Variable
from tests.core.common import (
    BATCH_SIZES_TO_TEST,
    check_another_theseus_function_is_copy,
)
from theseus.utils import numeric_jacobian


def test_gp_motion_model_cost_weight_weights():
    for dof in range(1, 10):
        for batch_size in BATCH_SIZES_TO_TEST:
            aux = torch.randn(batch_size, dof, dof).double()
            q_inv = aux.transpose(-2, -1).bmm(aux)
            dt = torch.rand(1).double()
            cost_weight = th.eb.GPCostWeight(q_inv, dt)
            sqrt_weights = cost_weight._compute_cost_weight()
            a = 12 * (dt.item() ** -3)
            b = -6 * (dt.item() ** -2)
            c = 4 * (dt.item() ** -1)
            weights = sqrt_weights.transpose(-2, -1).bmm(sqrt_weights)
            assert torch.allclose(weights[:, :dof, :dof], q_inv * a)
            assert torch.allclose(weights[:, :dof, dof:], q_inv * b)
            assert torch.allclose(weights[:, dof:, :dof], q_inv * b)
            assert torch.allclose(weights[:, dof:, dof:], q_inv * c)

            error = torch.randn(batch_size, 2 * dof).double()
            weighted_error = cost_weight.weight_error(error)
            assert torch.allclose(
                sqrt_weights @ error.view(batch_size, 2 * dof, 1),
                weighted_error.view(batch_size, 2 * dof, 1),
            )

            jacobians = [
                torch.randn(batch_size, 2 * dof, nvar).double() for nvar in range(1, 4)
            ]
            weighted_jacs, weighted_err_2 = cost_weight.weight_jacobians_and_error(
                jacobians, error
            )
            for i, jac in enumerate(jacobians):
                assert torch.allclose(weighted_jacs[i], sqrt_weights @ jacobians[i])
                assert torch.allclose(weighted_err_2, weighted_error)


def test_gp_motion_model_cost_weight_copy():
    q_inv = torch.randn(10, 2, 2)
    dt = torch.rand(1)
    cost_weight = th.eb.GPCostWeight(q_inv, dt, name="gp")
    check_another_theseus_function_is_copy(
        cost_weight, cost_weight.copy(new_name="new_name"), new_name="new_name"
    )
    check_another_theseus_function_is_copy(
        cost_weight, copy.deepcopy(cost_weight), new_name="gp_copy"
    )


def test_gp_motion_model_variable_type():
    for dof in range(1, 10):
        for batch_size in BATCH_SIZES_TO_TEST:
            aux = torch.randn(batch_size, dof, dof).double()
            q_inv = aux.transpose(-2, -1).bmm(aux)
            dt = torch.rand(1).double()
            cost_weight = th.eb.GPCostWeight(q_inv, dt)

            assert isinstance(cost_weight.Qc_inv, Variable)
            assert isinstance(cost_weight.dt, Variable)
            assert torch.allclose(cost_weight.Qc_inv.tensor, q_inv)
            assert torch.allclose(cost_weight.dt.tensor, dt)

            q_inv_v = Variable(q_inv)
            dt_v = Variable(dt)
            cost_weight = th.eb.GPCostWeight(q_inv_v, dt_v)
            assert isinstance(cost_weight.dt, Variable)
            assert cost_weight.Qc_inv is q_inv_v
            assert cost_weight.dt is dt_v

            q_inv_v = Variable(q_inv)
            dt_f = torch.rand(1)
            cost_weight = th.eb.GPCostWeight(q_inv_v, dt_f)
            assert isinstance(cost_weight.dt, Variable)
            assert np.allclose(cost_weight.dt.tensor.item(), dt_f)
            assert len(cost_weight.dt.shape) == 2


def test_gp_motion_model_cost_function_error_vector_vars():
    for batch_size in BATCH_SIZES_TO_TEST:
        for dof in range(1, 10):
            vars = [
                th.Vector(tensor=torch.randn(batch_size, dof).double())
                for _ in range(4)
            ]
            dt = th.Variable(torch.rand(1).double())

            q_inv = torch.randn(batch_size, dof, dof).double()
            # won't be used for the test, but it's required by cost_function's constructor
            cost_weight = th.eb.GPCostWeight(q_inv, dt)
            cost_function = th.eb.GPMotionModel(
                vars[0], vars[1], vars[2], vars[3], dt, cost_weight
            )

            error = cost_function.error()
            assert torch.allclose(
                error[:, :dof],
                vars[2].tensor - (vars[0].tensor + vars[1].tensor * dt.tensor),
            )
            assert torch.allclose(error[:, dof:], vars[3].tensor - vars[1].tensor)

            def new_error_fn(new_vars):
                new_cost_function = th.eb.GPMotionModel(
                    new_vars[0], new_vars[1], new_vars[2], new_vars[3], dt, cost_weight
                )
                return th.Vector(tensor=new_cost_function.error())

            expected_jacs = numeric_jacobian(new_error_fn, vars, function_dim=2 * dof)
            jacobians, error_jac = cost_function.jacobians()
            error = cost_function.error()
            assert torch.allclose(error_jac, error)
            for i in range(4):
                assert torch.allclose(jacobians[i], expected_jacs[i], atol=1e-8)
