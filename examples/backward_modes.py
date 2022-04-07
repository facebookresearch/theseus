#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This example illustrates the three backward modes (FULL, IMPLICIT, and TRUNCATED)
# on a problem fitting a quadratic to data.

from functools import partial

import torch
import theseus as th

import numpy as np
import numdifftools as nd

from collections import defaultdict
import time

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
    est = a.data * x.data.square() + b.data
    err = y.data - est
    return err


## DLM needs to add a new cost function that is just the optim_vars.
## We can give this a very small coefficient so that it doesn't affect the optimization too much.
## But this coefficient cannot be too small (we'll need to divide by this later on).

## We don't actually even need to add this to the forward pass, and just use a
## very small coefficient in the backward pass. It'll just be a slightly biased gradient.
# def _dlm_l2reg_fwd(optim_vars, aux_vars, coef=0.0001):
#     a, b = optim_vars
#     out = coef * a.data
#     return out


def _dlm_l2reg_bwd(optim_vars, aux_vars, eps=0.0001, grad_a=1.0):
    a, b = optim_vars
    out = eps * a.data - 1 / 2 * grad_a
    return out


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
## Add the regularization cost function.
# reg_function = th.AutoDiffCostFunction(
#     optim_vars,  # type: ignore
#     _dlm_l2reg_fwd,
#     1,
#     aux_vars=aux_vars,
#     name="_dlm_l2reg_fwd",
# )
objective = th.Objective()
objective.add(cost_function)
# objective.add(reg_function)
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
    optimizer_kwargs={
        "track_best_solution": True,
        "verbose": False,
        "backward_mode": th.BackwardMode.FULL,
    },
)

# The quadratic \hat y is now fit and we can also use Theseus
# to obtain the adjoint derivatives of \hat a with respect
# to other inputs or hyper-parameters, such as the data itself.
# Here we compute the derivative of \hat a with respect to the data,
# i.e. \partial a / \partial x using the full backward mode.
da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()

print("--- backward_mode=FULL")
print(da_dx.numpy())


print("--- backward_mode=DLM")
EPS = 0.00001

err = objective.error_squared_norm()
grad_x = torch.autograd.grad(err.sum(), data_x)[0]

## For DLM,
## we want to solve a perturbed optimization problem, then take the difference
## of gradients wrt the existing solution and the perturbed solution.

## Ideally want to use objective.copy() but I had some problems
## adding a new cost function that depends on new
## optim_vars and aux_vars.

bwd_objective = objective
# bwd_objective.erase("_dlm_l2reg_fwd")
completing_the_sq = th.AutoDiffCostFunction(
    # are these in the right order?
    list(bwd_objective.optim_vars.values()),
    partial(_dlm_l2reg_bwd, eps=EPS),
    1,
    aux_vars=aux_vars,
    name="_dlm_l2reg_bwd",
)
bwd_objective.add(completing_the_sq)
bwd_optimizer = th.GaussNewton(
    bwd_objective,
    max_iterations=15,  # don't really need this many iterations.
    step_size=0.5,
)
bwd_theseus_optim = th.TheseusLayer(bwd_optimizer)
bwd_theseus_inputs = {
    "a": updated_inputs["a"].detach().requires_grad_(),
    "b": updated_inputs["b"].detach().requires_grad_(),
    "x": data_x,
    "y": data_y,
}
bwd_updated_inputs, bwd_info = bwd_theseus_optim.forward(
    bwd_theseus_inputs,
    optimizer_kwargs={
        "track_best_solution": True,
        "verbose": False,
        "backward_mode": th.BackwardMode.FULL,
    },
)
err = bwd_objective.error_squared_norm()
grad_x_direct = torch.autograd.grad(err.sum(), data_x)[0]

da_dx = (grad_x - grad_x_direct) / EPS
print(da_dx.numpy())

# # We can also compute this using implicit differentiation by calling
# # forward again and changing the backward_mode flag.
# updated_inputs, info = theseus_optim.forward(
#     theseus_inputs,
#     optimizer_kwargs={
#         "track_best_solution": True,
#         "verbose": False,
#         "backward_mode": th.BackwardMode.IMPLICIT,
#     },
# )

# da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()
# print("\n--- backward_mode=IMPLICIT")
# print(da_dx.numpy())

# # We can also use truncated unrolling to compute the derivative:
# updated_inputs, info = theseus_optim.forward(
#     theseus_inputs,
#     optimizer_kwargs={
#         "track_best_solution": True,
#         "verbose": False,
#         "backward_mode": th.BackwardMode.TRUNCATED,
#         "backward_num_iterations": 5,
#     },
# )

# da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[0].squeeze()

# print("\n--- backward_mode=TRUNCATED, backward_num_iterations=5")
# print(da_dx.numpy())


# # Next we numerically check the derivative
# def fit_x(data_x_np):
#     theseus_inputs["x"] = (
#         torch.from_numpy(data_x_np).float().clone().requires_grad_().unsqueeze(0)
#     )
#     updated_inputs, _ = theseus_optim.forward(
#         theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose": False}
#     )
#     return updated_inputs["a"].item()


# data_x_np = data_x.detach().clone().numpy()
# dfit_x = nd.Gradient(fit_x)
# g = dfit_x(data_x_np)

# print("\n--- Numeric derivative")
# print(g)

# theseus_inputs["x"] = data_x

# # Next we run 10 trials of these computations and report the runtime
# # of the forward and backward passes.
# n_trials = 10
# times = defaultdict(list)
# for trial in range(n_trials + 1):
#     start = time.time()
#     updated_inputs, info = theseus_optim.forward(
#         theseus_inputs,
#         optimizer_kwargs={
#             "track_best_solution": True,
#             "verbose": False,
#             "backward_mode": th.BackwardMode.FULL,
#         },
#     )
#     times["fwd"].append(time.time() - start)

#     start = time.time()
#     da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
#         0
#     ].squeeze()
#     times["bwd"].append(time.time() - start)

#     updated_inputs, info = theseus_optim.forward(
#         theseus_inputs,
#         optimizer_kwargs={
#             "track_best_solution": True,
#             "verbose": False,
#             "backward_mode": th.BackwardMode.IMPLICIT,
#         },
#     )
#     start = time.time()
#     da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
#         0
#     ].squeeze()
#     times["bwd_impl"].append(time.time() - start)

#     updated_inputs, info = theseus_optim.forward(
#         theseus_inputs,
#         optimizer_kwargs={
#             "track_best_solution": True,
#             "verbose": False,
#             "backward_mode": th.BackwardMode.TRUNCATED,
#             "backward_num_iterations": 5,
#         },
#     )
#     start = time.time()
#     da_dx = torch.autograd.grad(updated_inputs["a"], data_x, retain_graph=True)[
#         0
#     ].squeeze()
#     times["bwd_trunc"].append(time.time() - start)


# print("\n=== Runtimes")
# k = "fwd"
# print(f"Forward: {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

# k = "bwd"
# print(f"Backward (FULL): {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

# k = "bwd_impl"
# print(f"Backward (IMPLICIT) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s")

# k = "bwd_trunc"
# print(
#     f"Backward (TRUNCATED, 5 steps) {np.mean(times[k]):.2e} s +/- {np.std(times[k]):.2e} s"
# )
