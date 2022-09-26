# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.core.cost_function import AutogradMode
from theseus.core.cost_weight import ScaleCostWeight

from .common import (
    MockCostFunction,
    MockCostWeight,
    MockVar,
    check_another_theseus_function_is_copy,
    create_mock_cost_functions,
)


def test_copy():
    cost_weight = MockCostWeight(th.Variable(torch.ones(1)))
    data = torch.ones(1, 1)
    cost_functions, *_ = create_mock_cost_functions(
        tensor=data, cost_weight=cost_weight
    )

    for cost_function in cost_functions:
        cost_function.weight = cost_weight

        def _check_new_cost_function(new_cost_function):
            check_another_theseus_function_is_copy(
                cost_function, new_cost_function, new_name=f"{cost_function.name}_copy"
            )
            check_another_theseus_function_is_copy(
                cost_weight,
                new_cost_function.weight,
                new_name=f"{cost_weight.name}_copy",
            )

        _check_new_cost_function(cost_function.copy())
        _check_new_cost_function(copy.deepcopy(cost_function))


def test_default_name_and_ids():
    reps = 100
    seen_ids = set()
    for i in range(reps):
        cost_function = MockCostFunction([], [], MockCostWeight(torch.ones(1)))
        cost_function_name = f"MockCostFunction__{cost_function._id}"
        seen_ids.add(cost_function._id)
        assert cost_function.name == cost_function_name
    assert len(seen_ids) == reps


# Adding three formatting options to include coverage for autograd mode resolution
@pytest.mark.parametrize("autograd_mode", ["DENSE", "loop_batch", AutogradMode.VMAP])
def test_autodiff_cost_function_error_and_jacobians_shape(autograd_mode):
    rng = torch.Generator()
    rng.manual_seed(0)
    for i in range(100):
        num_optim_vars = np.random.randint(0, 5)
        num_aux_vars = np.random.randint(0, 5)
        batch_size = np.random.randint(1, 10)
        err_dim = np.random.randint(1, 5)
        optim_vars = []
        aux_vars = []
        variable_values = torch.randn(num_optim_vars + num_aux_vars, generator=rng)
        idx = 0
        for i in range(num_optim_vars):
            optim_vars.append(
                MockVar(
                    idx + 1,
                    tensor=torch.ones(batch_size, idx + 1) * variable_values[idx],
                    name=f"optim_var_{i}",
                )
            )
            idx += 1
        for i in range(num_aux_vars):
            aux_vars.append(
                MockVar(
                    idx + 1,
                    tensor=torch.ones(batch_size, idx + 1) * variable_values[idx],
                    name=f"aux_var_{i}",
                )
            )
            idx += 1
        cost_weight = MockCostWeight(torch.ones(1, 1))

        # checks that the right number of optimization variables is passed
        # checks that the variable values are correct
        # returns the sum of the first elements of each tensor, which should be the
        # same as the sum of variables_values
        def error_fn(optim_vars, aux_vars):
            assert isinstance(optim_vars, tuple)
            assert len(optim_vars) == num_optim_vars
            assert len(optim_vars) > 0
            assert len(aux_vars) == num_aux_vars
            ret_val = torch.zeros(optim_vars[0].shape[0], err_dim)

            all_vars = optim_vars + aux_vars

            for i, arg in enumerate(all_vars):
                assert isinstance(arg, th.Variable)
                assert arg.shape == (batch_size, i + 1) or arg.shape == (1, i + 1)
                if autograd_mode != AutogradMode.VMAP:
                    assert arg.tensor.allclose(
                        variable_values[i] * torch.ones_like(arg.tensor)
                    )
                ret_val = ret_val + arg.tensor.view(arg.shape[0], -1).mean(
                    dim=1, keepdim=True
                )
            return ret_val

        # this checks that 0 optimization variables is not allowed
        if len(optim_vars) < 1:
            with pytest.raises(ValueError):
                th.AutoDiffCostFunction(
                    optim_vars,
                    error_fn,
                    1,
                    cost_weight=cost_weight,
                    aux_vars=aux_vars,
                )
        else:
            # check that the error function returns the correct value
            cost_function = th.AutoDiffCostFunction(
                optim_vars,
                error_fn,
                err_dim,
                cost_weight=cost_weight,
                aux_vars=aux_vars,
                autograd_mode=autograd_mode,
            )
            err = cost_function.error()
            assert err.allclose(
                variable_values.sum() * torch.ones(batch_size, err_dim), atol=1e-7
            )

            # Now checking the jacobians
            jacobians, err_jac = cost_function.jacobians()
            assert err_jac.allclose(err)
            assert len(jacobians) == num_optim_vars
            for i in range(num_optim_vars):
                # variable dim is i + 1 (see MockVar creation line)
                assert jacobians[i].shape == (batch_size, err_dim, i + 1)


@pytest.mark.parametrize(
    "autograd_mode", [AutogradMode.DENSE, AutogradMode.LOOP_BATCH, AutogradMode.VMAP]
)
def test_autodiff_cost_function_cost_weight(autograd_mode):
    batch_size = 10
    optim_vars = []
    aux_vars = []

    for i in range(5):
        optim_vars.append(
            MockVar(
                1,
                tensor=torch.ones(batch_size, 1) * torch.randn(1),
                name=f"optim_var_{i}",
            )
        )
        aux_vars.append(
            MockVar(
                1,
                tensor=torch.ones(batch_size, 1) * torch.randn(1),
                name=f"aux_var_{i}",
            )
        )

    def error_fn(optim_vars, aux_vars):
        assert len(optim_vars) > 0
        return torch.ones(optim_vars[0].shape[0], 1)

    # test verifying default CostWeight
    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        error_fn,
        1,
        aux_vars=aux_vars,
        autograd_mode=autograd_mode,
    )
    assert isinstance(cost_function.weight, ScaleCostWeight)
    assert torch.allclose(cost_function.weight.scale.tensor, torch.ones(1, 1))
    weighted_error = cost_function.weighted_error()
    assert torch.allclose(weighted_error, torch.ones(batch_size, 1))

    # test overriding default CostWeight
    for i in range(10):
        cost_weight_value = torch.randn(1, 1)
        cost_weight = MockCostWeight(cost_weight_value)
        cost_function = th.AutoDiffCostFunction(
            optim_vars,
            error_fn,
            1,
            cost_weight=cost_weight,
            aux_vars=aux_vars,
        )
        assert cost_function.weight is cost_weight
        assert torch.allclose(
            cost_function.weight.the_data.tensor, cost_weight_value
        )  # type: ignore
        weighted_error = cost_function.weighted_error()
        direct_error_computation = cost_weight_value * torch.ones(batch_size, 1)
        assert torch.allclose(weighted_error, direct_error_computation)


@pytest.mark.parametrize(
    "autograd_mode", [AutogradMode.DENSE, AutogradMode.LOOP_BATCH, AutogradMode.VMAP]
)
def test_autodiff_cost_function_to(autograd_mode):
    batch_size = 10
    optim_vars = []
    aux_vars = []

    for i in range(5):
        optim_vars.append(
            MockVar(
                1,
                tensor=torch.ones(batch_size, 1) * torch.randn(1),
                name=f"optim_var_{i}",
            )
        )
        aux_vars.append(
            MockVar(
                1,
                tensor=torch.ones(batch_size, 1) * torch.randn(1),
                name=f"aux_var_{i}",
            )
        )

    def error_fn(optim_vars, aux_vars):
        res = 0
        for var in optim_vars:
            res += var.tensor
        return res

    # test verifying default CostWeight
    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        error_fn,
        1,
        aux_vars=aux_vars,
        autograd_mode=autograd_mode,
    )

    for var in optim_vars:
        var.to(dtype=torch.double)

    # This fails because internal vars of the cost function have not been converted
    # to double
    with pytest.raises(ValueError):
        cost_function.jacobians()

    cost_function.to(dtype=torch.double)
    cost_function.jacobians()


@pytest.mark.parametrize(
    "autograd_mode", [AutogradMode.DENSE, AutogradMode.LOOP_BATCH, AutogradMode.VMAP]
)
def test_autodiff_cost_function_error_and_jacobians_shape_on_SO3(autograd_mode):
    for i in range(100):
        num_vars = np.random.randint(0, 5)
        batch_size = np.random.randint(1, 10)
        err_dim = 3
        optim_vars = []
        aux_vars = []
        idx = 0
        for i in range(num_vars):
            optim_vars.append(th.SO3.rand(batch_size, dtype=torch.float64))
            idx += 1
        for i in range(num_vars):
            aux_vars.append(th.Point3.rand(batch_size, dtype=torch.float64))
            idx += 1
        cost_weight = MockCostWeight(torch.ones(1, 1))

        def error_fn(optim_vars, aux_vars):
            assert isinstance(optim_vars, tuple)
            assert len(optim_vars) == num_vars
            assert len(optim_vars) > 0
            assert len(aux_vars) == num_vars
            ret_val = torch.zeros(optim_vars[0].shape[0], err_dim)

            for optim_var, aux_var in zip(optim_vars, aux_vars):
                ret_val = (
                    ret_val + th.SO3(tensor=optim_var.tensor).rotate(aux_var).tensor
                )

            return ret_val

        # this checks that 0 optimization variables is not allowed
        if len(optim_vars) < 1:
            with pytest.raises(ValueError):
                th.AutoDiffCostFunction(
                    optim_vars,
                    error_fn,
                    1,
                    cost_weight=cost_weight,
                    aux_vars=aux_vars,
                )
        else:
            # check that the error function returns the correct value
            cost_function = th.AutoDiffCostFunction(
                optim_vars,
                error_fn,
                err_dim,
                cost_weight=cost_weight,
                aux_vars=aux_vars,
                autograd_mode=autograd_mode,
            )
            err = cost_function.error()

            # Now checking the jacobians
            jacobians, err_jac = cost_function.jacobians()
            assert err_jac.allclose(err)
            assert len(jacobians) == num_vars
            for i in range(num_vars):
                # variable dim is i + 1 (see MockVar creation line)
                assert jacobians[i].shape == (batch_size, err_dim, 3)


@pytest.mark.parametrize(
    "autograd_mode", [AutogradMode.DENSE, AutogradMode.LOOP_BATCH, AutogradMode.VMAP]
)
def test_autodiff_cost_function_error_and_jacobians_value_on_SO3(autograd_mode):
    for i in range(100):
        num_vars = np.random.randint(0, 5)
        batch_size = np.random.randint(1, 10)
        err_dim = 3
        optim_vars = []
        aux_vars = []
        idx = 0
        for i in range(num_vars):
            optim_vars.append(th.SO3.rand(batch_size, dtype=torch.float64))
            idx += 1
        for i in range(num_vars):
            aux_vars.append(th.Point3.rand(batch_size, dtype=torch.float64))
            idx += 1
        cost_weight = MockCostWeight(torch.ones(1, 1, dtype=torch.float64))

        def error_fn(optim_vars, aux_vars):
            assert isinstance(optim_vars, tuple)
            assert len(optim_vars) == num_vars
            assert len(optim_vars) > 0
            assert len(aux_vars) == num_vars
            ret_val = torch.zeros(optim_vars[0].shape[0], err_dim, dtype=torch.float64)

            for optim_var, aux_var in zip(optim_vars, aux_vars):
                ret_val = (
                    ret_val + th.SO3(tensor=optim_var.tensor).rotate(aux_var).tensor
                )

            return ret_val

        # this checks that 0 optimization variables is not allowed
        if len(optim_vars) < 1:
            with pytest.raises(ValueError):
                th.AutoDiffCostFunction(
                    optim_vars,
                    error_fn,
                    1,
                    cost_weight=cost_weight,
                    aux_vars=aux_vars,
                )
        else:
            # check that the error function returns the correct value
            cost_function = th.AutoDiffCostFunction(
                optim_vars,
                error_fn,
                err_dim,
                cost_weight=cost_weight,
                aux_vars=aux_vars,
                autograd_mode=autograd_mode,
            )
            jac_actual, err_actual = cost_function.jacobians()

            err_expected = torch.zeros(batch_size, 3, dtype=torch.float64)
            for n in torch.arange(num_vars):
                jac = []  # type: ignore
                err_expected += optim_vars[n].rotate(aux_vars[n], jacobians=jac).tensor
                assert torch.allclose(jac_actual[n], jac[0])

            assert torch.allclose(err_actual, err_expected)
