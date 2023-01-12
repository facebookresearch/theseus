# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

import torch

import theseus as th
from tests.core.common import BATCH_SIZES_TO_TEST


def _new_robust_cf(
    batch_size, loss_cls, generator, masked_weight=False
) -> th.RobustCostFunction:
    v1 = th.rand_se3(batch_size, generator=generator)
    v2 = th.rand_se3(batch_size, generator=generator)
    if masked_weight:
        mask = torch.randint(2, (batch_size, 1), generator=generator).bool()
        assert mask.any()
        assert not mask.all()
        w_tensor = torch.randn(batch_size, 1, generator=generator) * mask
    else:
        w_tensor = torch.randn(1, generator=generator)
    w = th.ScaleCostWeight(w_tensor)
    cf = th.Local(v1, v2, w)
    ll_radius = th.Variable(tensor=torch.randn(1, 1, generator=generator))
    return th.RobustCostFunction(cf, loss_cls, ll_radius)


def _grad(jac: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    return jac.transpose(2, 1).matmul(error.unsqueeze(2)).permute(0, 2, 1)


def test_robust_cost_weighted_error():
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(10):
        for batch_size in BATCH_SIZES_TO_TEST:
            for loss_cls in [th.WelschLoss, th.HuberLoss]:
                robust_cf = _new_robust_cf(batch_size, loss_cls, generator)
                cf = robust_cf.cost_function
                e = cf.weighted_error()
                rho = robust_cf.weighted_error()
                rho2 = (rho * rho).sum(dim=1, keepdim=True)
                # `RobustCostFunction.weighted_error` is written so that
                # ||we||2 == rho(||e||2)
                expected_rho2 = loss_cls.evaluate(
                    (e * e).sum(dim=1, keepdim=True), robust_cf.log_loss_radius.tensor
                )
                torch.testing.assert_close(rho2, expected_rho2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("loss_cls", [th.WelschLoss, th.HuberLoss])
def test_robust_cost_grad_form(batch_size, loss_cls):
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(10):
        robust_cf = _new_robust_cf(batch_size, loss_cls, generator)
        cf = robust_cf.cost_function
        jacs, e = cf.weighted_jacobians_error()
        cf_grad = _grad(jacs[0], e)
        e_norm = (e * e).sum(1, keepdim=True)
        rho_prime = loss_cls.linearize(e_norm, robust_cf.log_loss_radius.tensor)
        # `weighted_jacobians_error()` is written so that it results in a
        # gradient equal to drho_de2 * J^T * e, which in the code is
        # `rho_prime * cf_grad`.
        expected_grad = rho_prime.view(-1, 1, 1) * cf_grad
        rescaled_jac, rescaled_e = robust_cf.weighted_jacobians_error()
        grad = _grad(rescaled_jac[0], rescaled_e)
        torch.testing.assert_close(grad, expected_grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("loss_cls", [th.WelschLoss, th.HuberLoss])
def test_robust_cost_jacobians(batch_size, loss_cls):
    generator = torch.Generator()
    generator.manual_seed(0)

    for _ in range(10):
        robust_cf = _new_robust_cf(batch_size, loss_cls, generator)
        v1, v2 = robust_cf.cost_function.var, robust_cf.cost_function.target
        v_aux = v1.copy()
        ll_radius = robust_cf.log_loss_radius
        w = robust_cf.cost_function.weight

        def test_fn(v_data):
            v_aux.update(v_data)
            new_robust_cf = th.RobustCostFunction(
                th.Local(v_aux, v2, w), loss_cls, ll_radius
            )
            e = new_robust_cf.cost_function.weighted_error()
            e_norm = (e * e).sum(1, keepdim=True)
            return loss_cls.evaluate(e_norm, ll_radius.tensor) / 2.0

        aux_id = torch.arange(batch_size)
        grad_raw_dense = torch.autograd.functional.jacobian(test_fn, (v1.tensor,))[0]
        grad_raw_sparse = grad_raw_dense[aux_id, :, aux_id]
        expected_grad = v1.project(grad_raw_sparse, is_sparse=True)

        rescaled_jac, rescaled_err = robust_cf.weighted_jacobians_error()
        grad = _grad(rescaled_jac[0], rescaled_err)

        torch.testing.assert_close(grad, expected_grad, atol=1e-2, rtol=1e-2)


def test_masked_jacobians_called(monkeypatch):
    rng = torch.Generator()
    rng.manual_seed(0)
    robust_cf = _new_robust_cf(128, th.WelschLoss, rng, masked_weight=True)
    robust_cf._supports_masking = True

    called = [False]

    def masked_jacobians_mock(cost_fn, mask):
        called[0] = True
        return cost_fn.jacobians()

    monkeypatch.setattr(
        th.core.cost_function, "masked_jacobians", masked_jacobians_mock
    )
    robust_cf.weighted_jacobians_error()
    assert called[0]


@pytest.mark.parametrize("loss_cls", [th.WelschLoss, th.HuberLoss])
def test_mask_jacobians(loss_cls):
    batch_size = 512
    rng = torch.Generator()
    rng.manual_seed(0)
    robust_cf = _new_robust_cf(batch_size, loss_cls, rng, masked_weight=True)
    jac_expected, err_expected = robust_cf.weighted_jacobians_error()
    robust_cf._supports_masking = True
    jac, err = robust_cf.weighted_jacobians_error()
    torch.testing.assert_close(err, err_expected)
    for j1, j2 in zip(jac, jac_expected):
        torch.testing.assert_close(j1, j2)
