# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random

import pytest  # noqa: F401
import torch

import theseus as th
from tests.core.common import MockCostFunction, MockCostWeight, MockVar


def test_default_variable_ordering():
    # repeat test a few times with different inputs
    for _ in range(5):
        # vary the number of variables to have in the objective
        for num_variables in range(2, 11):
            # generate all possible 2-variable cost functions, then shuffle their add order
            variables = []
            for i in range(num_variables):
                variables.append(MockVar(1, tensor=None, name=f"var{i}"))
            variable_pairs = [c for c in itertools.combinations(variables, 2)]
            random.shuffle(variable_pairs)

            # add the cost function to the objective and store the order of variable addition
            expected_variable_order = []
            objective = th.Objective()
            cost_weight = MockCostWeight(
                th.Variable(torch.ones(1), name="cost_weight_aux")
            )
            for var1, var2 in variable_pairs:
                cost_function = MockCostFunction([var1, var2], [], cost_weight)
                if var1 not in expected_variable_order:
                    expected_variable_order.append(var1)
                if var2 not in expected_variable_order:
                    expected_variable_order.append(var2)
                objective.add(cost_function)

            # check the the default variable order matches the expected order
            default_order = th.VariableOrdering(objective)
            for i, var in enumerate(expected_variable_order):
                assert i == default_order.index_of(var.name)


def test_variable_ordering_append_and_remove():
    variables = [MockVar(1, tensor=None, name=f"var{i}") for i in range(50)]
    mock_objective = th.Objective()
    mock_objective.optim_vars = dict([(var.name, var) for var in variables])
    # repeat a bunch of times with different order
    for _ in range(100):
        random.shuffle(variables)
        order = th.VariableOrdering(mock_objective, default_order=False)
        for v in variables:
            order.append(v)
        for i, v in enumerate(variables):
            assert i == order.index_of(v.name)
            assert v == order[i]
        assert order.complete

        random.shuffle(variables)
        for v in variables:
            order.remove(v)
            assert not order.complete
            assert v not in order._var_order
            assert v.name not in order._var_name_to_index


def test_variable_ordering_iterator():
    variables = [MockVar(1, tensor=None, name=f"var{i}") for i in range(50)]
    mock_objective = th.Objective()
    mock_objective.optim_vars = dict([(var.name, var) for var in variables])
    order = th.VariableOrdering(mock_objective, default_order=False)
    for v in variables:
        order.append(v)

    i = 0
    for v in order:
        assert v == variables[i]
        i += 1
