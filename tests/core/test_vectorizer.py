# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from unittest import mock

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from tests.core.common import BATCH_SIZES_TO_TEST
from theseus.core.vectorizer import _CostFunctionWrapper


def test_costs_vars_and_err_before_vectorization():
    for _ in range(20):
        objective = th.Objective()
        batch_size = torch.randint(low=1, high=10, size=(1,)).item()
        v1 = th.Vector(tensor=torch.randn(batch_size, 1), name="v1")
        v2 = th.Vector(tensor=torch.randn(batch_size, 1), name="v2")
        odummy = th.Vector(1, name="odummy")
        t1 = th.Vector(tensor=torch.zeros(1, 1), name="t1")
        adummy = th.Variable(tensor=torch.zeros(1, 1), name="adummy")
        cw1 = th.ScaleCostWeight(th.Variable(torch.zeros(1, 1), name="w1"))
        cw2 = th.ScaleCostWeight(th.Variable(torch.zeros(1, 1), name="w2"))
        cf1 = th.Difference(v1, t1, cw1)

        # Also test with autodiff cost
        def err_fn(optim_vars, aux_vars):
            return optim_vars[0].tensor - aux_vars[0].tensor

        cf2 = th.AutoDiffCostFunction([v2, odummy], err_fn, 1, cw2, [t1, adummy])

        # Chech that vectorizer's has the correct number of wrappers
        objective.add(cf1)
        objective.add(cf2)
        th.Vectorize(objective)

        # Update weights after creating vectorizer to see if data is picked up correctly
        w1 = torch.randn(1, 1)  # also check that broadcasting works
        w2 = torch.randn(batch_size, 1)

        # disable for this test since we are not checking the result
        objective._vectorization_needs_update = lambda: False
        objective.update({"w1": w1, "w2": w2})

        def _check_attr(cf, var):
            return hasattr(cf, var.name) and getattr(cf, var.name) is var

        # Check that the vectorizer's cost functions have the right variables and error
        saw_cf1 = False
        saw_cf2 = False
        for cf in objective._get_jacobians_iter():
            assert isinstance(cf, _CostFunctionWrapper)
            optim_vars = [v for v in cf.optim_vars]
            aux_vars = [v for v in cf.aux_vars]
            assert t1 in aux_vars
            assert _check_attr(cf, t1)
            w_err = cf.weighted_error()
            if cf.cost_fn is cf1:
                assert v1 in optim_vars
                assert w_err.allclose((v1.tensor - t1.tensor) * w1)
                assert _check_attr(cf, v1)
                saw_cf1 = True
            elif cf.cost_fn is cf2:
                assert v2 in optim_vars and odummy in optim_vars
                assert adummy in aux_vars
                assert _check_attr(cf, v2) and _check_attr(cf, odummy)
                assert w_err.allclose((v2.tensor - t1.tensor) * w2)
                saw_cf2 = True
            else:
                assert False
        assert saw_cf1 and saw_cf2


def test_correct_schemas_and_shared_vars():
    v1 = th.Vector(1)
    v2 = th.Vector(1)
    tv = th.Vector(1)
    w1 = th.ScaleCostWeight(1.0)
    mv = th.Vector(1)

    v3 = th.Vector(3)
    v4 = th.Vector(3)

    s1 = th.SE2()
    s2 = th.SE2()
    ts = th.SE2()

    objective = th.Objective()
    # these two can be grouped
    cf1 = th.Difference(v1, tv, w1)
    cf2 = th.Difference(v2, tv, w1)
    objective.add(cf1)
    objective.add(cf2)

    # this one uses the same weight and v1, v2, but cannot be grouped
    cf3 = th.Between(v1, v2, mv, w1)
    objective.add(cf3)

    # this one is the same cost function type, var type, and weight but different
    # dimension, so cannot be grouped either
    cf4 = th.Difference(v3, v4, w1)
    objective.add(cf4)

    # Now add another group with a different data-type (no-shared weight)
    w2 = th.ScaleCostWeight(1.0)
    w3 = th.ScaleCostWeight(2.0)
    cf5 = th.Difference(s1, ts, w2)
    cf6 = th.Difference(s2, ts, w3)
    objective.add(cf5)
    objective.add(cf6)

    # Not grouped with anything cf1 and cf2 because weight type is different
    w7 = th.DiagonalCostWeight([[1.0]])
    cf7 = th.Difference(v1, tv, w7)
    objective.add(cf7)

    vectorization = th.Vectorize(objective)

    assert len(vectorization._schema_dict) == 5
    seen_cnt = [0] * 7
    for schema, cost_fn_wrappers in vectorization._schema_dict.items():
        cost_fns = [w.cost_fn for w in cost_fn_wrappers]
        var_names = vectorization._var_names[schema]
        if cf1 in cost_fns:
            assert len(cost_fns) == 2
            assert cf2 in cost_fns
            seen_cnt[0] += 1
            seen_cnt[1] += 1
            assert f"{th.Vectorize._SHARED_TOKEN}{w1.scale.name}" in var_names
            assert f"{th.Vectorize._SHARED_TOKEN}{tv.name}" in var_names
        if cf3 in cost_fns:
            assert len(cost_fns) == 1
            seen_cnt[2] += 1
        if cf4 in cost_fns:
            assert len(cost_fns) == 1
            seen_cnt[3] += 1
        if cf5 in cost_fns:
            assert len(cost_fns) == 2
            assert cf6 in cost_fns
            seen_cnt[4] += 1
            seen_cnt[5] += 1
            assert f"{th.Vectorize._SHARED_TOKEN}{w2.scale.name}" not in var_names
            assert f"{th.Vectorize._SHARED_TOKEN}{w3.scale.name}" not in var_names
            assert f"{th.Vectorize._SHARED_TOKEN}{ts.name}" in var_names
        if cf7 in cost_fns:
            assert len(cost_fns) == 1
            seen_cnt[6] += 1
    assert seen_cnt == [1] * 7


def test_correct_schemas_for_autodiffcosts():
    v1_1 = th.Vector(1)
    v1_2 = th.Vector(1)
    v1_3 = th.Vector(1)
    v2_1 = th.Vector(2)
    v2_2 = th.Vector(2)

    objective = th.Objective()

    def err_fn_1(optim_vars, aux_vars):
        return

    def err_fn_2(optim_vars, aux_vars):
        return

    # these two can be grouped
    cf1 = th.AutoDiffCostFunction([v1_1, v1_2], err_fn_1, 1, aux_vars=[v2_1])
    cf2 = th.AutoDiffCostFunction([v1_1, v1_3], err_fn_1, 1, aux_vars=[v2_2])
    objective.add(cf1)
    objective.add(cf2)

    # these two can be grouped, but the group is different because of the err fn
    cf3 = th.AutoDiffCostFunction([v1_2, v1_3], err_fn_2, 1, aux_vars=[v2_1])
    cf4 = th.AutoDiffCostFunction([v1_1, v1_3], err_fn_2, 1, aux_vars=[v2_2])
    objective.add(cf3)
    objective.add(cf4)

    vectorization = th.Vectorize(objective)

    assert len(vectorization._schema_dict) == 2
    seen_cnt = [0] * 4
    for schema, cost_fn_wrappers in vectorization._schema_dict.items():
        cost_fns = [w.cost_fn for w in cost_fn_wrappers]
        vectorized_cost = vectorization._vectorized_cost_fns[schema]
        assert isinstance(vectorized_cost, th.AutoDiffCostFunction)
        if cf1 in cost_fns:
            assert len(cost_fns) == 2
            assert cf2 in cost_fns
            assert vectorized_cost._err_fn is err_fn_1
            seen_cnt[0] += 1
            seen_cnt[1] += 1
        if cf3 in cost_fns:
            assert len(cost_fns) == 2
            assert cf4 in cost_fns
            assert vectorized_cost._err_fn is err_fn_2
            seen_cnt[2] += 1
            seen_cnt[3] += 1
    assert seen_cnt == [1] * 4


def _check_vectorized_wrappers(vectorization, objective):
    for w in vectorization._cost_fn_wrappers:
        for cost_fn in objective.cost_functions.values():
            if cost_fn is w.cost_fn:
                w_jac, w_err = cost_fn.weighted_jacobians_error()
                assert w._cached_error.allclose(w_err)
                for jac, exp_jac in zip(w._cached_jacobians, w_jac):
                    torch.testing.assert_close(jac, exp_jac, atol=1e-6, rtol=1e-6)


def test_vectorized_error():
    rng = np.random.default_rng(0)
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(20):
        dim = rng.choice([1, 2])
        objective = th.Objective()
        batch_size = rng.choice(range(1, 11))

        n_vars = rng.choice([1, 10])
        vectors = [
            th.Vector(
                tensor=torch.randn(batch_size, dim, generator=generator), name=f"v{i}"
            )
            for i in range(n_vars)
        ]
        target = th.Vector(dim, name="target")
        w = th.ScaleCostWeight(torch.randn(1, generator=generator))
        for v in vectors:
            objective.add(th.Difference(v, target, w))

        se3s = [
            th.SE3(
                tensor=th.SE3.rand(batch_size, generator=generator).tensor,
                strict=False,
            )
            for i in range(rng.choice([1, 10]))
        ]
        s_target = th.SE3.rand(1, generator=generator)
        ws = th.DiagonalCostWeight(torch.randn(1, 6, generator=generator))
        for s in se3s:
            objective.add(th.Difference(s, s_target, ws))

        vectorization = th.Vectorize(objective)
        objective.update_vectorization_if_needed()

        assert objective._vectorized_jacobians_iter is vectorization._cost_fn_wrappers
        _check_vectorized_wrappers(vectorization, objective)

        squared_error = torch.cat(
            [cf.weighted_error() for cf in objective.cost_functions.values()], dim=1
        )
        torch.testing.assert_close(squared_error, objective.error())

        # Check that calling error(also_update=False) changes the result but doesn't
        # change the vectorized wrappers
        another_squared_error = objective.error(
            {f"v{i}": torch.ones(batch_size, dim) for i in range(n_vars)},
            also_update=False,
        )
        assert not another_squared_error.allclose(squared_error)
        _check_vectorized_wrappers(vectorization, objective)

        # Just to be sure, check that objective.error() is calling
        # the vectorized error iterator.
        called = [False]

        def mock_err_iter(self):
            called[0] = True
            return self._cost_fn_wrappers  # just to have the same return type

        with mock.patch.object(
            th.Vectorize, "_get_vectorized_error_iter", mock_err_iter
        ):
            th.Vectorize(objective)
            objective.error()
            assert called[0]


def test_vectorized_retract():
    rng = np.random.default_rng(0)
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(100):
        variables = []
        deltas = []
        batch_size = rng.choice(range(1, 11))
        n_vars = rng.choice(BATCH_SIZES_TO_TEST)
        for _ in range(n_vars):
            var_type: th.LieGroup = rng.choice(
                [th.Vector, th.SE2, th.SE3, th.SO2, th.SO3]
            )
            if var_type == th.Vector:
                dof = rng.integers(1, 10)
                var = th.Vector.rand(batch_size, dof, generator=generator)
            else:
                var = var_type.rand(batch_size, generator=generator)
            deltas.append(torch.randn((batch_size, var.dof()), generator=generator))
            variables.append(var)
        variables_vectorized = [v.copy() for v in variables]
        delta = torch.cat(deltas, dim=1)

        ignore_mask = torch.rand(batch_size, generator=generator) > 0.5
        force_update = rng.random() > 0.5
        th.Objective._retract_base(
            delta, variables, ignore_mask=ignore_mask, force_update=force_update
        )
        th.Vectorize._vectorized_retract_optim_vars(
            delta,
            variables_vectorized,
            ignore_mask=ignore_mask,
            force_update=force_update,
        )

        for v1, v2 in zip(variables, variables_vectorized):
            assert v1.tensor.allclose(v2.tensor)


# This solves a very simple objective of the form sum (wi * (xi - ti)) **2, where
# some wi can be zero with some probability. When vectorize=True, our vectorization
# class will compute masked batched jacobians. So, this function can be used to test
# that the solution is the same when this feature is on/off. We also check if we
# can do a backward pass when this masking is used.
def _solve_fn_for_masked_jacobians(
    batch_size, dof, num_costs, weight_cls, vectorize, device
):
    rng = torch.Generator()
    rng.manual_seed(batch_size)
    obj = th.Objective()
    variables = [th.Vector(dof=dof, name=f"x{i}") for i in range(num_costs)]
    targets = [
        th.Vector(tensor=torch.randn(batch_size, dof, generator=rng), name=f"t{i}")
        for i in range(num_costs)
    ]
    base_tensor = torch.ones(
        batch_size, dof if weight_cls == th.DiagonalCostWeight else 1, device=device
    )
    # Wrapped into a param to pass to torch optimizer if necessary
    params = [
        torch.nn.Parameter(
            base_tensor.clone() * (torch.rand(1, generator=rng).item() > 0.9)
        )
        for _ in range(num_costs)
    ]
    weights = [weight_cls(params[i]) for i in range(num_costs)]
    for i in range(num_costs):
        obj.add(th.Difference(variables[i], targets[i], weights[i], name=f"cf{i}"))

    input_tensors = {
        f"x{i}": torch.ones(batch_size, dof, device=device) for i in range(num_costs)
    }
    layer = th.TheseusLayer(
        th.LevenbergMarquardt(obj, step_size=0.1, max_iterations=5),
        vectorize=vectorize,
    )
    layer.to(device=device)
    sol, _ = layer.forward(input_tensors)

    # Check that we can backprop through this without errors
    if vectorize:
        optim = torch.optim.Adam(params, lr=1e-4)
        for _ in range(5):  # do a few steps
            optim.zero_grad()
            layer.forward(input_tensors)
            loss = obj.error_metric().sum()
            loss.backward()
            optim.step()

    return sol


@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("dof", [1, 4])
@pytest.mark.parametrize("num_costs", [1, 64])
@pytest.mark.parametrize("weight_cls", [th.ScaleCostWeight, th.DiagonalCostWeight])
def test_masked_jacobians(batch_size, dof, num_costs, weight_cls):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sol1 = _solve_fn_for_masked_jacobians(
        batch_size, dof, num_costs, weight_cls, True, device
    )
    sol2 = _solve_fn_for_masked_jacobians(
        batch_size, dof, num_costs, weight_cls, False, device
    )

    for i in range(num_costs):
        torch.testing.assert_close(sol1[f"x{i}"], sol2[f"x{i}"])


def test_masked_jacobians_called(monkeypatch):
    called = [False]

    def masked_jacobians_mock(cost_fn, mask):
        called[0] = True
        return cost_fn.jacobians()

    monkeypatch.setattr(
        th.core.cost_function, "masked_jacobians", masked_jacobians_mock
    )

    _solve_fn_for_masked_jacobians(128, 2, 16, th.ScaleCostWeight, True, "cpu")
    assert called[0]
