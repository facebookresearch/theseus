#!/usr/bin/env python3
#
# This example illustrates the three backward modes (FULL, IMPLICIT, and TRUNCATED)
# on a problem fitting a quadratic to data.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import theseus as th
import theseus.optimizer.nonlinear as thnl

import numpy as np
import numdifftools as nd

from collections import defaultdict
import time

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
    est = a.data * x.data.square() + b.data
    err = y.data - est
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
    step_size=0.5,
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
    track_best_solution=True,
    verbose=False,
    backward_mode=thnl.BackwardMode.FULL,
)

da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()

print("--- backward_mode=FULL")
print(da_dx.numpy())


updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
    track_best_solution=True,
    verbose=False,
    backward_mode=thnl.BackwardMode.IMPLICIT,
)

da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()
print("\n--- backward_mode=IMPLICIT")
print(da_dx.numpy())

updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
    track_best_solution=True,
    verbose=False,
    backward_mode=thnl.BackwardMode.TRUNCATED,
    backward_num_iterations=5,
)

da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()

print("\n--- backward_mode=TRUNCATED, backward_num_iterations=5")
print(da_dx.numpy())


def fit_x(data_x_np):
    theseus_inputs["x"] = (
        torch.from_numpy(data_x_np).float().clone().requires_grad_().unsqueeze(0)
    )
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, track_best_solution=True, verbose=False
    )
    return updated_inputs["a"].item()


data_x_np = data_x.detach().clone().numpy()
dfit_x = nd.Gradient(fit_x)
g = dfit_x(data_x_np)

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
        track_best_solution=True,
        verbose=False,
        backward_mode=thnl.BackwardMode.FULL,
    )
    times["fwd"].append(time.time() - start)

    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd"].append(time.time() - start)

    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        track_best_solution=True,
        verbose=False,
        backward_mode=thnl.BackwardMode.IMPLICIT,
    )
    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd_impl"].append(time.time() - start)

    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        track_best_solution=True,
        verbose=False,
        backward_mode=thnl.BackwardMode.TRUNCATED,
        backward_num_iterations=5,
    )
    start = time.time()
    da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
        0
    ].squeeze()
    times["bwd_trunc"].append(time.time() - start)


print("\n=== Runtimes")
k = "fwd"
print(f"Forward: {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

k = "bwd"
print(f"Backward (FULL): {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

k = "bwd_impl"
print(f"Backward (IMPLICIT) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

k = "bwd_trunc"
print(
    f"Backward (TRUNCATED, 5 steps) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s"
)
