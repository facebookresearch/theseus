#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This example illustrates the four backward modes
# (unroll, implicit, truncated, and dlm)
# on a problem fitting a quadratic to data.

import time
from collections import defaultdict

import numpy as np
import torch

import theseus as th

torch.manual_seed(0)


# Sample from a quadratic y = ax^2 + b*noise
def generate_data(num_points=10, a=1.0, b=0.5, noise_factor=0.01):
    data_x = torch.rand((1, num_points))
    noise = torch.randn((1, num_points)) * noise_factor
    data_y = a * data_x.square() + b + noise
    return data_x, data_y


num_points = 10
data_x, data_y = generate_data(num_points)
x = th.Variable(data_x.requires_grad_(), name="x")
y = th.Variable(data_y.requires_grad_(), name="y")

# We now attempt to recover the quadratic from the data with
# theseus by formulating it as a non-linear least squares
# optimization problem.
# We write the model as \hat y = \hat a x^2 + \hat b,
# where the parameters \hat a and \hat b are just `a` and `b`
# in the code here.
a = th.Vector(1, name="a")
b = th.Vector(1, name="b")


# The error is y - \hat y
def quad_error_fn(optim_vars, aux_vars):
    a, b = optim_vars
    x, y = aux_vars
    est = a.tensor * x.tensor.square() + b.tensor
    err = y.tensor - est
    return err


# We then use Theseus to optimize \hat a and \hat b so that
# y = \hat y for all datapoints
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
updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
    optimizer_kwargs={
        "track_best_solution": True,
        "verbose": False,
        "backward_mode": "unroll",
    },
)

# The quadratic \hat y is now fit and we can also use Theseus
# to obtain the adjoint derivatives of \hat a with respect
# to other inputs or hyper-parameters, such as the data itself.
# Here we compute the derivative of \hat a with respect to the data,
# i.e. \partial a / \partial x using the unroll backward mode.
da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()

print("--- backward_mode=unroll")
print(da_dx.numpy())

# We can also compute this using implicit differentiation by calling
# forward again and changing the backward_mode flag.
updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
    optimizer_kwargs={
        "track_best_solution": True,
        "verbose": False,
        "backward_mode": "implicit",
    },
)

da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()
print("\n--- backward_mode=implicit")
print(da_dx.numpy())

# We can also use truncated unrolling to compute the derivative:
updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
    optimizer_kwargs={
        "track_best_solution": True,
        "verbose": False,
        "backward_mode": "truncated",
        "backward_num_iterations": 5,
    },
)

da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()

print("\n--- backward_mode=truncated, backward_num_iterations=5")
print(da_dx.numpy())


# We can also compute the direct loss minimization gradient.
updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
    optimizer_kwargs={
        "track_best_solution": True,
        "verbose": False,
        "backward_mode": "dlm",
        "dlm_epsilon": 1e-3,
    },
)

da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()
print("\n--- backward_mode=dlm")
print(da_dx.numpy())


# Next we numerically check the derivative
with torch.no_grad():

    def fn(data_x_torch):
        theseus_inputs["x"] = data_x_torch
        updated_inputs, _ = theseus_optim.forward(
            theseus_inputs,
            optimizer_kwargs={"track_best_solution": True, "verbose": False},
        )
        return updated_inputs["a"]

    g = (
        torch.autograd.functional.jacobian(fn, data_x.detach())[0, 0, 0]
        .double()
        .numpy()
    )
print("\n--- Numeric derivative")
print(g)

theseus_inputs["x"] = data_x

# Next we run 10 trials of these computations and report the runtime
# of the forward and backward passes.
n_trials = 10
times = defaultdict(list)
for trial in range(n_trials + 1):
    start = time.time()
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "unroll",
        },
    )
    times["fwd"].append(time.time() - start)

    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd"].append(time.time() - start)

    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "implicit",
        },
    )
    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd_impl"].append(time.time() - start)

    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "truncated",
            "backward_num_iterations": 5,
        },
    )
    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd_trunc"].append(time.time() - start)

    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={
            "track_best_solution": True,
            "verbose": False,
            "backward_mode": "dlm",
            "dlm_epsilon": 1e-3,
        },
    )
    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd_dlm"].append(time.time() - start)


print("\n=== Runtimes")
k = "fwd"
print(f"Forward: {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

k = "bwd"
print(f"Backward (unroll): {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

k = "bwd_impl"
print(f"Backward (implicit) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

k = "bwd_trunc"
print(
    f"Backward (truncated, 5 steps) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s"
)

k = "bwd_dlm"
print(f"Backward (dlm) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")
