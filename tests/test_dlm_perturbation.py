# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import theseus as th
from tests.core.common import BATCH_SIZES_TO_TEST
from theseus.theseus_layer import _DLMPerturbation
from theseus.utils import numeric_jacobian


def _original_dlm_perturbation(optim_vars, aux_vars):
    v = optim_vars[0]
    g = aux_vars[0]
    epsilon = aux_vars[1]
    return epsilon.tensor * v.tensor - 0.5 * g.tensor


def test_dlm_perturbation_jacobian():
    generator = torch.Generator()
    generator.manual_seed(0)
    rng = np.random.default_rng(0)
    dtype = torch.float64
    for _ in range(100):
        group_cls = rng.choice([th.Vector, th.SE3, th.SE2, th.SO2, th.SO3])
        for batch_size in BATCH_SIZES_TO_TEST:
            epsilon = th.Variable(
                tensor=torch.randn(batch_size, 1, dtype=dtype, generator=generator)
            )

            if group_cls == th.Vector:
                dof = rng.choice([1, 2])
                var = group_cls.randn(batch_size, dof, dtype=dtype, generator=generator)
                grad = group_cls.randn(
                    batch_size, dof, dtype=dtype, generator=generator
                )
            else:
                var = group_cls.randn(batch_size, dtype=dtype, generator=generator)
                grad = group_cls.randn(batch_size, dtype=dtype, generator=generator)

            w = th.ScaleCostWeight(1.0).to(dtype=dtype)
            cf = _DLMPerturbation(var, epsilon, grad, w)

            def new_error_fn(vars):
                new_cost_function = _DLMPerturbation(vars[0], epsilon, grad, w)
                return th.Vector(tensor=new_cost_function.error())

            expected_jacs = numeric_jacobian(
                new_error_fn,
                [var],
                function_dim=np.prod(var.shape[1:]),
                delta_mag=1e-6,
            )
            jacobians, error_jac = cf.jacobians()
            error = cf.error()
            assert error.allclose(error_jac)
            assert jacobians[0].allclose(expected_jacs[0], atol=1e-5)

            if group_cls in [th.Vector, th.SO2, th.SE2]:
                # Original cf didn't work for SO3 or SE3
                original_cf = th.AutoDiffCostFunction(
                    [var],
                    _original_dlm_perturbation,
                    var.shape[1],
                    aux_vars=[grad, epsilon],
                )
                original_jac, original_err = original_cf.jacobians()
                assert error.allclose(original_err)
                assert jacobians[0].allclose(original_jac[0], atol=1e-5)


def test_backward_pass_se3_runs():
    generator = torch.Generator()
    generator.manual_seed(0)
    dtype = torch.float64
    batch_size = 10
    var = th.rand_se3(batch_size, generator=generator)
    var.name = "v1"
    target = th.rand_se3(batch_size, generator=generator)
    target.name = "target"

    objective = th.Objective()
    objective.add(th.Difference(var, target, th.ScaleCostWeight(1.0)))
    objective.to(dtype=dtype)
    optimizer = th.GaussNewton(objective)
    layer = th.TheseusLayer(optimizer)

    target_data = torch.nn.Parameter(th.rand_se3(batch_size, dtype=dtype).tensor)
    adam = torch.optim.Adam([target_data], lr=0.01)
    loss0 = None
    for _ in range(5):
        adam.zero_grad()
        with th.enable_lie_tangent(), th.no_lie_group_check(silent=True):
            out, _ = layer.forward(
                {"target": target_data},
                optimizer_kwargs={
                    "backward_mode": "dlm",
                    "verbose": False,
                },
            )

            loss = out["v1"].norm()
            if loss0 is None:
                loss0 = loss.item()
            loss.backward()
            adam.step()
    assert loss.item() < loss0
