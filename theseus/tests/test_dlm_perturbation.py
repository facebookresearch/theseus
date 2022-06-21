import numpy as np
import torch

import theseus as th
from theseus.theseus_layer import _DLMPerturbation
from theseus.utils import numeric_jacobian


def test_dlm_perturbation_jacobian():
    generator = torch.Generator()
    generator.manual_seed(0)
    rng = np.random.default_rng(0)
    dtype = torch.float64
    for _ in range(100):
        group_cls = rng.choice([th.Vector, th.SE3, th.SE2, th.SO2, th.SO3])
        for batch_size in [1, 10, 100]:
            epsilon = th.Variable(
                data=torch.randn(batch_size, 1, dtype=dtype, generator=generator)
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
                return th.Vector(data=new_cost_function.error())

            expected_jacs = numeric_jacobian(
                new_error_fn,
                [var],
                function_dim=np.prod(var.shape[1:]),
                delta_mag=1e-6,
            )
            jacobians, error_jac = cf.jacobians()
            error = cf.error()
            assert torch.allclose(error_jac, error)
            assert torch.allclose(jacobians[0], expected_jacs[0], atol=1e-5)
