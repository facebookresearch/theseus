# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th

torch.manual_seed(0)


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


def quad_error_fn(optim_vars, aux_vars):
    a, b = optim_vars
    x, y = aux_vars
    est = a.tensor * x.tensor.square() + b.tensor
    err = y.tensor - est
    return err


optim_vars = [a, b]
aux_vars = [x, y]
cost_function = th.AutoDiffCostFunction(
    optim_vars,  # type: ignore
    quad_error_fn,
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
)

theseus_inputs = {
    "a": 2 * torch.ones((1, 1)).requires_grad_(),
    "b": torch.ones((1, 1)).requires_grad_(),
    "x": data_x,
    "y": data_y,
}
theseus_optim = th.TheseusLayer(optimizer)


def test_backwards():
    # First we use torch.autograd.functional to numerically compute the gradient
    # the optimal a w.r.t. the x part of the data
    with torch.no_grad():

        def fn(data_x_torch):
            theseus_inputs["x"] = data_x_torch
            updated_inputs, _ = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={"track_best_solution": True, "verbose": False},
            )
            return updated_inputs["a"]

        da_dx_numeric = torch.autograd.functional.jacobian(fn, data_x.detach())[0, 0, 0]

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
    assert torch.allclose(da_dx_numeric, da_dx_unroll, atol=1e-3)

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
    assert torch.allclose(da_dx_numeric, da_dx_implicit, atol=1e-4)

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
    assert torch.allclose(da_dx_numeric, da_dx_truncated, atol=1e-4)

    updated_inputs, _ = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": th.BackwardMode.DLM,
            "dlm_epsilon": 0.001,
        },
    )
    da_dx_truncated = torch.autograd.grad(
        updated_inputs["a"], data_x, retain_graph=True
    )[0].squeeze()
    assert torch.allclose(da_dx_numeric, da_dx_truncated, atol=1e-3)
