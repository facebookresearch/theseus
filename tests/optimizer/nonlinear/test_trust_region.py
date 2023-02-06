# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

import theseus as th
import torch


# Passing different linear_solver_cls to test with both dense/sparse linearization
# Also test sparse [cpu/gpu] because they go through different custom backends
@pytest.mark.parametrize("dof", [1, 8])
@pytest.mark.parametrize(
    "linear_solver_cls",
    [th.CholeskyDenseSolver, th.CholmodSparseSolver, th.LUCudaSparseSolver],
)
def test_rho(dof, linear_solver_cls):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    if device == "cpu" and linear_solver_cls == th.LUCudaSparseSolver:
        return
    if device == "cuda:0" and linear_solver_cls == th.CholmodSparseSolver:
        return

    def _rand_v():
        return torch.randn(1, dof, generator=rng, device=device)

    def _rand_w():
        return th.ScaleCostWeight(torch.randn(1, generator=rng, device=device))

    for dof in [1, 8]:
        n_vars = 16
        # This test checks that rho = 1 for a simple linear problem:
        #  min sum (xi - ti) ** 2 + ((x[i+-i] - x[i]) - m[i]) ** 2
        vs = [th.Vector(tensor=_rand_v(), name=f"x{i}") for i in range(n_vars)]
        o = th.Objective()
        for i in range(n_vars):
            t = th.Vector(tensor=_rand_v(), name=f"t{i}")
            o.add(th.Difference(vs[i], t, _rand_w(), name=f"diff{i}"))
            if i > 0:
                m = th.Vector(tensor=_rand_v(), name=f"m{i}")
                o.add(th.Between(vs[i], vs[i - 1], m, _rand_w(), name=f"btw{i}"))

        o.to(device=device)
        # This is testing TrustRegion base class rather than Dogleg's
        # Using Dogleg because it's the only subclass of TrustRegion atm
        opt = th.Dogleg(o, linear_solver_cls=linear_solver_cls)
        o._resolve_batch_size()
        opt.linear_solver.linearization.linearize()
        previous_err = opt.objective.error_metric()

        # Check rho = 1. Predicted error by TrustRegion method should
        # match actual error after step for a linear problem
        for _ in range(100):
            delta = torch.randn(1, dof * n_vars, device=device, generator=rng)
            _, new_err = opt._compute_retracted_tensors_and_error(
                delta, torch.zeros_like(delta[:, 0]), False
            )
            rho = opt._compute_rho(delta, previous_err, new_err)
            torch.testing.assert_close(rho, torch.ones_like(rho), atol=1e-3, rtol=1e-3)
