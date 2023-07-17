# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

import theseus as th

import torch
from tests.theseus_tests.core.common import BATCH_SIZES_TO_TEST


def _new_robust_cf(
    batch_size,
    loss_cls,
    generator,
    masked_weight=False,
    gnc_cost=False,
) -> [th.RobustCostFunction, th.GNCRobustCostFunction]:
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
    if gnc_cost:
        gnc_control_val = th.Variable(
            tensor=torch.randn(1, 1, generator=generator).abs() * 1e3
        )
        return th.GNCRobustCostFunction(
            cf,
            loss_cls,
            log_loss_radius=ll_radius,
            gnc_control_val=gnc_control_val,
        )
    else:
        return th.RobustCostFunction(cf, loss_cls, log_loss_radius=ll_radius)


def _grad(jac: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    return jac.transpose(2, 1).matmul(error.unsqueeze(2)).permute(0, 2, 1)


def _loss_evaluate(robust_cf, loss_cls, x: torch.Tensor) -> torch.Tensor:
    if isinstance(robust_cf, th.GNCRobustCostFunction):
        return loss_cls.evaluate(
            x,
            log_radius=robust_cf.log_loss_radius.tensor,
            mu=robust_cf.gnc_control_val.tensor,
        )
    else:
        return loss_cls.evaluate(x, log_radius=robust_cf.log_loss_radius.tensor)


def _loss_linearize(robust_cf, loss_cls, x: torch.Tensor) -> torch.Tensor:
    if isinstance(robust_cf, th.GNCRobustCostFunction):
        return loss_cls.linearize(
            x,
            log_radius=robust_cf.log_loss_radius.tensor,
            mu=robust_cf.gnc_control_val.tensor,
        )
    else:
        return loss_cls.linearize(x, log_radius=robust_cf.log_loss_radius.tensor)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize(
    "loss_cls", [th.WelschLoss, th.HingeLoss, th.HuberLoss, th.GemanMcClureLoss]
)
def test_robust_cost_weighted_error(batch_size, loss_cls):
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(10):
        robust_cf = _new_robust_cf(
            batch_size,
            loss_cls,
            generator,
            gnc_cost=issubclass(loss_cls, th.GemanMcClureLoss),
        )
        cf = robust_cf.cost_function
        e = cf.weighted_error()
        rho = robust_cf.weighted_error()
        rho2 = (rho * rho).sum(dim=1, keepdim=True)
        # `RobustCostFunction.weighted_error` is written so that
        # ||we||2 == rho(||e||2)
        expected_rho2 = _loss_evaluate(
            robust_cf,
            loss_cls,
            (e * e).sum(dim=1, keepdim=True),
        )
        torch.testing.assert_close(rho2, expected_rho2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize(
    "loss_cls", [th.WelschLoss, th.HingeLoss, th.HuberLoss, th.GemanMcClureLoss]
)
def test_robust_cost_grad_form(batch_size, loss_cls):
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(10):
        robust_cf = _new_robust_cf(
            batch_size,
            loss_cls,
            generator,
            gnc_cost=issubclass(loss_cls, th.GemanMcClureLoss),
        )
        cf = robust_cf.cost_function
        jacs, e = cf.weighted_jacobians_error()
        cf_grad = _grad(jacs[0], e)
        e_norm = (e * e).sum(1, keepdim=True)
        rho_prime = _loss_linearize(robust_cf, loss_cls, e_norm)
        # `weighted_jacobians_error()` is written so that it results in a
        # gradient equal to drho_de2 * J^T * e, which in the code is
        # `rho_prime * cf_grad`.
        expected_grad = rho_prime.view(-1, 1, 1) * cf_grad
        rescaled_jac, rescaled_e = robust_cf.weighted_jacobians_error()
        grad = _grad(rescaled_jac[0], rescaled_e)
        torch.testing.assert_close(grad, expected_grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize(
    "loss_cls", [th.WelschLoss, th.HingeLoss, th.HuberLoss, th.GemanMcClureLoss]
)
def test_robust_cost_jacobians(batch_size, loss_cls):
    generator = torch.Generator()
    generator.manual_seed(0)

    for _ in range(10):
        robust_cf = _new_robust_cf(
            batch_size,
            loss_cls,
            generator,
            gnc_cost=issubclass(loss_cls, th.GemanMcClureLoss),
        )
        v1, v2 = robust_cf.cost_function.var, robust_cf.cost_function.target
        v_aux = v1.copy()
        ll_radius = robust_cf.log_loss_radius
        w = robust_cf.cost_function.weight

        def test_fn(v_data):
            v_aux.update(v_data)
            new_robust_cf = (
                th.GNCRobustCostFunction(
                    th.Local(v_aux, v2, w),
                    loss_cls,
                    log_loss_radius=ll_radius,
                    gnc_control_val=robust_cf.gnc_control_val,
                )
                if issubclass(loss_cls, th.GemanMcClureLoss)
                else th.RobustCostFunction(
                    th.Local(v_aux, v2, w), loss_cls, log_loss_radius=ll_radius
                )
            )
            e = new_robust_cf.cost_function.weighted_error()
            e_norm = (e * e).sum(1, keepdim=True)
            return (
                _loss_evaluate(
                    new_robust_cf,
                    loss_cls,
                    e_norm,
                )
                / 2.0
            )

        aux_id = torch.arange(batch_size)
        grad_raw_dense = torch.autograd.functional.jacobian(test_fn, (v1.tensor,))[0]
        grad_raw_sparse = grad_raw_dense[aux_id, :, aux_id]
        expected_grad = v1.project(grad_raw_sparse, is_sparse=True)

        rescaled_jac, rescaled_err = robust_cf.weighted_jacobians_error()
        grad = _grad(rescaled_jac[0], rescaled_err)

        torch.testing.assert_close(grad, expected_grad, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "loss_cls", [th.WelschLoss, th.HingeLoss, th.HuberLoss, th.GemanMcClureLoss]
)
def test_masked_jacobians_called(monkeypatch, loss_cls):
    rng = torch.Generator()
    rng.manual_seed(0)
    robust_cf = _new_robust_cf(
        128,
        loss_cls,
        rng,
        masked_weight=True,
        gnc_cost=issubclass(loss_cls, th.GemanMcClureLoss),
    )
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


@pytest.mark.parametrize(
    "loss_cls", [th.WelschLoss, th.HingeLoss, th.HuberLoss, th.GemanMcClureLoss]
)
def test_mask_jacobians(loss_cls):
    batch_size = 512
    rng = torch.Generator()
    rng.manual_seed(0)
    robust_cf = _new_robust_cf(
        batch_size,
        loss_cls,
        rng,
        masked_weight=True,
        gnc_cost=issubclass(loss_cls, th.GemanMcClureLoss),
    )
    jac_expected, err_expected = robust_cf.weighted_jacobians_error()
    robust_cf._supports_masking = True
    jac, err = robust_cf.weighted_jacobians_error()
    torch.testing.assert_close(err, err_expected)
    for j1, j2 in zip(jac, jac_expected):
        torch.testing.assert_close(j1, j2)


def _data_model(a, b, x):
    return a * x.square() + b


def _generate_data(num_points=100, a=1, b=0.5, noise_factor=0.01):
    data_x = torch.rand((1, num_points))
    noise = torch.randn((1, num_points)) * noise_factor
    return data_x, _data_model(a, b, data_x) + noise


@pytest.mark.parametrize("batch_size", [1, 4])
def test_flatten_dims(batch_size):
    # This creates two objectives for a regression problem
    # and compares their linearization
    #   - Obj1: N 1d robust costs functions each evaluating one residual term
    #   - Obj2: 1 Nd robust cost function that evaluates all residuals at once
    # Data for regression problem y ~ Normal(Ax^2 + B, sigma)
    n = 10
    data_x, data_y = _generate_data(num_points=n)
    data_y[:, :5] = 1000  # include some extreme outliers for robust cost

    # optimization variables are of type Vector with 2 degrees of freedom (dof)
    # one for A and one for B
    ab = th.Vector(2, name="ab")

    def residual_fn(optim_vars, aux_vars):
        ab = optim_vars[0]
        x, y = aux_vars
        return y.tensor - _data_model(ab.tensor[:, :1], ab.tensor[:, 1:], x.tensor)

    w = th.ScaleCostWeight(0.5)
    log_loss_radius = th.as_variable(0.5)

    # First create an objective with individual cost functions per error term
    # Need individual aux variables to represent data of each residual terms
    xs = [th.Vector(1, name=f"x{i}") for i in range(n)]
    ys = [th.Vector(1, name=f"y{i}") for i in range(n)]
    obj_unrolled = th.Objective()
    for i in range(n):
        obj_unrolled.add(
            th.RobustCostFunction(
                th.AutoDiffCostFunction(
                    (ab,), residual_fn, 1, aux_vars=(xs[i], ys[i]), cost_weight=w
                ),
                th.HuberLoss,
                log_loss_radius,
                name=f"rcf{i}",
            )
        )
    lin_unrolled = th.DenseLinearization(obj_unrolled)
    th_inputs = {f"x{i}": data_x[:, i].view(1, 1) for i in range(data_x.shape[1])}
    th_inputs.update({f"y{i}": data_y[:, i].view(1, 1) for i in range(data_y.shape[1])})
    th_inputs.update({"ab": torch.rand((batch_size, 2))})
    obj_unrolled.update(th_inputs)
    lin_unrolled.linearize()

    # Now one with a single vectorized cost function, and flatten_dims=True
    # Residual terms call all be represented with "batched" data variables
    xb = th.Vector(n, name="xb")
    yb = th.Vector(n, name="yb")
    obj_flattened = th.Objective()
    obj_flattened.add(
        th.RobustCostFunction(
            th.AutoDiffCostFunction(
                (ab,), residual_fn, n, aux_vars=(xb, yb), cost_weight=w
            ),
            th.HuberLoss,
            log_loss_radius,
            name="rcf",
            flatten_dims=True,
        )
    )
    lin_flattened = th.DenseLinearization(obj_flattened)
    th_inputs = {
        "xb": data_x,
        "yb": data_y,
        "ab": th_inputs["ab"],  # reuse the previous random value
    }
    obj_flattened.update(th_inputs)
    lin_flattened.linearize()

    # Both objectives should result in the same error and linearizations
    torch.testing.assert_close(obj_unrolled.error(), obj_flattened.error())
    torch.testing.assert_close(lin_unrolled.b, lin_flattened.b)
    torch.testing.assert_close(lin_unrolled.AtA, lin_flattened.AtA)
