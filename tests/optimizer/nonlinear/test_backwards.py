# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
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


def quad_sq_error_fn(optim_vars, aux_vars):
    a, b = optim_vars
    x, y = aux_vars
    est = a.tensor * x.tensor.square() + b.tensor
    err = (y.tensor - est) ** 2
    return err


def numeric_grad(f, h=1e-4):
    # Approximate the gradient with a central difference.
    def df(x):
        assert x.ndim == 1
        n = x.shape[0]
        g = np.zeros_like(x)
        for i in range(n):
            h_i = np.zeros_like(x)
            h_i[i] = h
            g[i] = (f(x + h_i) - f(x - h_i)) / (2.0 * h)
        return g

    return df


def test_backwards():
    for error_fn in [quad_error_fn, quad_sq_error_fn]:
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
            abs_err_tolerance=1e-15,
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
        da_dx_unroll = torch.autograd.grad(
            updated_inputs["a"], data_x, retain_graph=True
        )[0].squeeze()
        tol = 1e-3 if error_fn == quad_error_fn else 1e-1
        torch.testing.assert_close(da_dx_numeric, da_dx_unroll, atol=tol, rtol=tol)

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
        tol = 1e-3 if error_fn == quad_error_fn else 0.3
        torch.testing.assert_close(da_dx_numeric, da_dx_implicit, atol=tol, rtol=tol)

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
        tol = 1e-3 if error_fn == quad_error_fn else 1e-1
        torch.testing.assert_close(da_dx_numeric, da_dx_truncated, atol=tol, rtol=tol)

        if error_fn == quad_error_fn:
            updated_inputs, _ = theseus_optim.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": False,
                    "backward_mode": th.BackwardMode.DLM,
                    "dlm_epsilon": 0.0001,
                },
            )
            da_dx_truncated = torch.autograd.grad(
                updated_inputs["a"], data_x, retain_graph=True
            )[0].squeeze()
            torch.testing.assert_close(
                da_dx_numeric, da_dx_truncated, atol=1e-1, rtol=1e-1
            )
