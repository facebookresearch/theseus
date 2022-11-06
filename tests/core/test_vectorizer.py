# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        for cf in objective._get_iterator():
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


def test_vectorized_error():
    rng = np.random.default_rng(0)
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(20):
        dim = rng.choice([1, 2])
        objective = th.Objective()
        batch_size = rng.choice(range(1, 11))

        vectors = [
            th.Vector(
                tensor=torch.randn(batch_size, dim, generator=generator), name=f"v{i}"
            )
            for i in range(rng.choice([1, 10]))
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

        assert objective._cost_functions_iterable is vectorization._cost_fn_wrappers
        for w in vectorization._cost_fn_wrappers:
            for cost_fn in objective.cost_functions.values():
                if cost_fn is w.cost_fn:
                    w_jac, w_err = cost_fn.weighted_jacobians_error()
                    assert w._cached_error.allclose(w_err)
                    for jac, exp_jac in zip(w._cached_jacobians, w_jac):
                        assert jac.allclose(exp_jac, atol=1e-6)

        squared_error = torch.cat(
            [cf.weighted_error() for cf in objective.cost_functions.values()], dim=1
        )
        assert squared_error.allclose(objective.error())


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
