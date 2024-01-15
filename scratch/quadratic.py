#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is a small test of the augmented Lagrangian solver for the
# optimization problem:
#     minimize ||x||^2 subject to x_0 = 1,
# which has the analytic solution [1, 0].

import torch
import theseus as th
from augmented_lagrangian import solve_augmented_lagrangian


def err_fn(optim_vars, aux_vars):
    (x,) = optim_vars
    cost = x.tensor
    return cost


def equality_constraint_fn(optim_vars, aux_vars):
    (x,) = optim_vars
    x = x.tensor
    return x[..., 0] - 1


dim = 2
x = th.Vector(dim, name="x")
optim_vars = [x]

state_dict = solve_augmented_lagrangian(
    err_fn=err_fn,
    optim_vars=optim_vars,
    aux_vars=[],
    dim=dim,
    equality_constraint_fn=equality_constraint_fn,
    optimizer_cls=th.LevenbergMarquardt,
    optimizer_kwargs=dict(max_iterations=100, step_size=1.0),
    verbose=True,
    initial_state_dict={},
)

true_solution = torch.tensor([1.0, 0.0])
threshold = 1e-5
assert (state_dict["x"].squeeze() - true_solution).norm() <= threshold
