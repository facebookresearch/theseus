# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

import theseus as th

from tests.theseus_tests.decorators import run_if_baspacho
from theseus.utils import numeric_grad

torch.manual_seed(0)


@pytest.mark.parametrize(
    "linear_solver_cls", [th.CholeskyDenseSolver, th.CholmodSparseSolver]
)
def test_backwards_quad_fit(linear_solver_cls):
    def generate_data(num_points=10, a=1.0, b=0.5, noise_factor=0.01):
        data_x = torch.rand((1, num_points))
        noise = torch.randn((1, num_points)) * noise_factor
        data_y = a * data_x.square() + b + noise
        return data_x, data_y

    num_points = 10
    data_x, data_y = generate_data(num_points)

    x = th.Variable(data_x.requires_grad_(), name="x")
    y = th.Variable(data_y.requires_grad_(), name="y")
    a = th.Vector(1, name="a")
    b = th.Vector(1, name="b")

    def error_fn(optim_vars, aux_vars):
        a, b = optim_vars
        x, y = aux_vars
        est = a.tensor * x.tensor.square() + b.tensor
        err = y.tensor - est
        return err

    optim_vars = [a, b]
    aux_vars = [x, y]
    cost_function = th.AutoDiffCostFunction(
        optim_vars,  # type: ignore
        error_fn,
        num_points,
        aux_vars=aux_vars,
        name="quadratic_cost_fn",
    )
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.GaussNewton(
        objective,
        max_iterations=15,
        step_size=1.0,
        linear_solver_cls=linear_solver_cls,
    )

    theseus_inputs = {
        "a": 2 * torch.ones((1, 1)).requires_grad_(),
        "b": torch.ones((1, 1)).requires_grad_(),
        "x": data_x,
        "y": data_y,
    }
    theseus_optim = th.TheseusLayer(optimizer)

    # First we use torch.autograd.functional to numerically compute the gradient
    # the optimal a w.r.t. the x part of the data
    with torch.no_grad():

        def fit_x(data_x_np):
            theseus_inputs["x"] = (
                torch.from_numpy(data_x_np)
                .float()
                .clone()
                .requires_grad_()
                .unsqueeze(0)
            )
            updated_inputs, _ = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={"track_best_solution": True, "verbose": False},
            )
            return updated_inputs["a"].item()

        data_x_np = data_x.detach().clone().numpy().squeeze()
        dfit_x = numeric_grad(fit_x, h=1e-4)
        da_dx_numeric = torch.from_numpy(dfit_x(data_x_np)).float()

    theseus_inputs["x"] = data_x
    updated_inputs, _ = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "unroll",
        },
    )
    da_dx_unroll = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    torch.testing.assert_close(da_dx_numeric, da_dx_unroll, atol=1e-3, rtol=1e-3)

    updated_inputs, _ = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": th.BackwardMode.IMPLICIT,
        },
    )
    da_dx_implicit = torch.autograd.grad(
        updated_inputs["a"], data_x, retain_graph=True
    )[0].squeeze()
    torch.testing.assert_close(da_dx_numeric, da_dx_implicit, atol=1e-3, rtol=1e-3)

    updated_inputs, _ = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "TRUNCATED",
            "backward_num_iterations": 5,
        },
    )
    da_dx_truncated = torch.autograd.grad(
        updated_inputs["a"], data_x, retain_graph=True
    )[0].squeeze()
    torch.testing.assert_close(da_dx_numeric, da_dx_truncated, atol=1e-3, rtol=1e-3)

    updated_inputs, _ = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": th.BackwardMode.DLM,
            "dlm_epsilon": 0.001,
        },
    )
    da_dx_dlm = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    torch.testing.assert_close(da_dx_numeric, da_dx_dlm, atol=1e-1, rtol=1e-1)


@run_if_baspacho()
@pytest.mark.parametrize(
    "linear_solver_cls",
    [
        th.CholeskyDenseSolver,
        th.CholmodSparseSolver,
        th.LUCudaSparseSolver,
        th.BaspachoSparseSolver,
    ],
)
def test_backwards_quartic(linear_solver_cls):
    device = "cpu"
    if linear_solver_cls in [
        th.LUCudaSparseSolver,
        th.BaspachoSparseSolver,
    ]:
        if not torch.cuda.is_available():
            return
        device = "cuda:0"

    def error_fn(optim_vars, aux_vars):
        (a,) = optim_vars
        (x,) = aux_vars
        err = (a.tensor - x.tensor) ** 2
        return err

    a = th.Vector(1, name="a")
    x_th = torch.zeros([1, 1]).requires_grad_()
    x = th.Variable(x_th, name="x")
    optim_vars = [a]
    aux_vars = [x]
    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        error_fn,
        1,
        aux_vars=aux_vars,
        name="cost_fn",
    )
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.GaussNewton(
        objective,
        max_iterations=15,
        step_size=1.0,
        linear_solver_cls=linear_solver_cls,
    )

    theseus_inputs = {
        "a": torch.ones([1, 1]).requires_grad_().to(device),
        "x": x_th.to(device),
    }
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(device)

    updated_inputs, _ = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "implicit",
        },
    )

    da_dx = torch.autograd.grad(updated_inputs["a"], x_th, retain_graph=True)[
        0
    ].squeeze()

    # Equality should hold exactly even in floating point
    # because of how the derivatives cancel
    assert da_dx.item() == 1.5


def test_implicit_fallback_linear_solver():
    # Create a singular system that can only be solved if damping added
    x = th.Vector(2, name="x")
    t = th.Vector(2, name="t")

    o = th.Objective()
    w = th.DiagonalCostWeight(torch.FloatTensor([1, 0]).view(1, 2))
    o.add(th.Difference(x, t, w, name="cost"))
    opt = th.TheseusLayer(th.LevenbergMarquardt(o, max_iterations=5))

    input_dict = {"x": torch.ones(1, 2), "t": torch.zeros(1, 2)}

    # __strict_implicit_final_gn__ = True shows that this problem leads to errors
    with pytest.raises(RuntimeError):
        opt.forward(
            input_dict,
            optimizer_kwargs={
                "damping": 0.1,
                "backward_mode": "implicit",
                "__strict_implicit_final_gn__": True,
            },
        )
    # No error is raised by default
    opt.forward(
        input_dict,
        optimizer_kwargs={"damping": 0.1, "backward_mode": "implicit"},
    )
