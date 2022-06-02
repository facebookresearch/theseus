import torch

import theseus as th
from theseus.core.vectorizer import _CostFunctionWrapper


def test_costs_vars_and_err_before_vectorization():
    for _ in range(20):
        objective = th.Objective()
        batch_size = torch.randint(low=1, high=10, size=(1,)).item()
        v1 = th.Vector(data=torch.randn(batch_size, 1), name="v1")
        v2 = th.Vector(data=torch.randn(batch_size, 1), name="v2")
        odummy = th.Vector(1, name="odummy")
        t1 = th.Vector(data=torch.zeros(batch_size, 1), name="t1")
        adummy = th.Variable(data=torch.zeros(1, 1), name="adummy")
        cw1 = th.ScaleCostWeight(th.Variable(torch.zeros(1, 1), name="w1"))
        cw2 = th.ScaleCostWeight(th.Variable(torch.zeros(1, 1), name="w2"))
        cf1 = th.Difference(v1, cw1, t1)

        # Also test with autodiff cost
        def err_fn(optim_vars, aux_vars):
            return optim_vars[0] - aux_vars[0]

        cf2 = th.AutoDiffCostFunction([v2, odummy], err_fn, 1, cw2, [t1, adummy])

        # Chech that vectorizer's has the correct number of wrappers
        objective.add(cf1)
        objective.add(cf2)
        th.Vectorize(objective)

        # Update weights after creating vectorizer to see if data is picked up correctly
        w1 = torch.randn(1, 1)  # also check that broadcasting works
        w2 = torch.randn(batch_size, 1)
        objective.update({"w1": w1, "w2": w2})

        def _check_attr(cf, var):
            return hasattr(cf, var.name) and getattr(cf, var.name) is var

        # Check that the vectorizer's cost functions have the right variables and error
        saw_cf1 = False
        saw_cf2 = False
        for cf in objective:
            assert isinstance(cf, _CostFunctionWrapper)
            optim_vars = [v for v in cf.optim_vars]
            aux_vars = [v for v in cf.aux_vars]
            assert t1 in aux_vars
            assert _check_attr(cf, t1)
            w_err = cf.weighted_error()
            if cf.cost_fn is cf1:
                assert v1 in optim_vars
                assert w_err.allclose((v1.data - t1.data) * w1)
                assert _check_attr(cf, v1)
                saw_cf1 = True
            elif cf.cost_fn is cf2:
                assert v2 in optim_vars and odummy in optim_vars
                assert adummy in aux_vars
                assert _check_attr(cf, v2) and _check_attr(cf, odummy)
                assert w_err.allclose((v2.data - t1.data) * w2)
                saw_cf2 = True
            else:
                assert False
        assert saw_cf1 and saw_cf2
