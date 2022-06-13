import torch

import theseus as th
from theseus.utils import numeric_jacobian


def _new_robust_cf(batch_size, loss_cls, generator) -> th.RobustCostFunction:
    v1 = th.rand_se3(batch_size)
    v2 = th.rand_se3(batch_size)
    w = th.ScaleCostWeight(torch.randn(1, generator=generator))
    cf = th.Local(v1, w, v2)
    ll_radius = th.Variable(data=torch.randn(1, 1, generator=generator))
    return th.RobustCostFunction(cf, loss_cls, ll_radius)


def _grad(jac: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    return jac.transpose(2, 1).matmul(error.unsqueeze(2)).permute(0, 2, 1)


def test_robust_cost_weighted_error():
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(10):
        for batch_size in [1, 2, 10]:
            for loss_cls in [th.WelschLoss, th.HuberLoss]:
                robust_cf = _new_robust_cf(batch_size, loss_cls, generator)
                cf = robust_cf.cost_function
                e = cf.weighted_error()
                rho = robust_cf.weighted_error()
                rho2 = (rho * rho).sum(dim=1, keepdim=True)
                # `RobustCostFunction.weighted_error` is written so that
                # ||we||2 == rho(||e||2)
                expected_rho2 = loss_cls.evaluate(
                    (e * e).sum(dim=1, keepdim=True), robust_cf.log_loss_radius.data
                )
                assert torch.allclose(rho2, expected_rho2)


def test_robust_cost_grad_form():
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(10):
        for batch_size in [1, 2, 10]:
            for loss_cls in [th.WelschLoss, th.HuberLoss]:
                robust_cf = _new_robust_cf(batch_size, loss_cls, generator)
                cf = robust_cf.cost_function
                jacs, e = cf.weighted_jacobians_error()
                cf_grad = _grad(jacs[0], e)
                e_norm = (e * e).sum(1, keepdim=True)
                rho_prime = loss_cls.linearize(e_norm, robust_cf.log_loss_radius.data)
                # `weighted_jacobians_error()` is written so that it results in a
                # gradient equal to drho_de2 * J^T * e, which in the code is
                # `rho_prime * cf_grad`.
                expected_grad = rho_prime.view(-1, 1, 1) * cf_grad
                rescaled_jac, rescaled_e = robust_cf.weighted_jacobians_error()
                grad = _grad(rescaled_jac[0], rescaled_e)
                assert torch.allclose(grad, expected_grad, atol=1e-6)


def test_robust_cost_jacobians():
    generator = torch.Generator()
    generator.manual_seed(0)

    v1 = th.rand_vector(3, 3)
    v2 = th.rand_vector(3, 3)
    w = th.ScaleCostWeight(torch.randn(1, generator=generator))
    cf = th.Local(v1, w, v2)
    ll_radius = th.Variable(data=torch.ones(1, 1))
    robust_cf = th.RobustCostFunction(cf, th.WelschLoss, ll_radius)
    loss_cls = th.WelschLoss

    def new_error_fn(vars):
        new_robust_cf = th.RobustCostFunction(
            th.Local(vars[0], w, v2), loss_cls, ll_radius
        )
        e = new_robust_cf.cost_function.weighted_error()
        e_norm = (e * e).sum(1, keepdim=True)
        rho = loss_cls.evaluate(e_norm, ll_radius.data)
        return th.Vector(data=rho)

    expected_grad = numeric_jacobian(
        new_error_fn,
        [v1],
        function_dim=1,
        delta_mag=1e-6,
    )[0]
    print(expected_grad)

    rescaled_jac, rescaled_err = robust_cf.weighted_jacobians_error()
    grad = _grad(rescaled_jac[0], rescaled_err)

    print(grad, expected_grad)
