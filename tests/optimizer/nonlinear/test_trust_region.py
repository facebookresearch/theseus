import pytest

import theseus as th
import torch


@pytest.mark.parametrize("dof", [1, 8])
def test_rho(dof):
    # This test checks that rho = 1 for a simple linear problem min (x - t) ** 2
    x = th.Vector(tensor=torch.zeros(1, dof), name="x")
    t = th.Vector(tensor=torch.ones(1, dof), name="t")
    o = th.Objective()
    o.add(th.Difference(x, t, th.ScaleCostWeight(1.0), name="cf"))
    # This is testing TrustRegion base class rather than Dogleg's
    # Using Dogleg because it's the only subclass of TrustRegion atm
    opt = th.Dogleg(o)
    o._resolve_batch_size()
    opt.linear_solver.linearization.linearize()
    previous_err = opt._error_metric()

    # Check rho = 1. Predicted error by TrustRegion method should
    # match actual error after step for a linear problem
    for _ in range(10):  # repeat a few times with random delta
        delta = torch.randn(1, dof)
        _, new_err = opt._compute_retracted_tensors_and_error(
            delta, torch.zeros_like(delta[:, 0]), False
        )
        rho = opt._compute_rho(delta, previous_err, new_err)
        torch.testing.assert_close(rho, torch.ones_like(rho))
