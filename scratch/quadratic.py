#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# import numpy as np
import torch

import theseus as th
import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

torch.manual_seed(0)


def g(x):
    # return x[..., 0] - 1
    return x - 1


def aug_lagrang(mu, z):
    # def g(optim_vars):
    #     (x,) = optim_vars
    #     return x.tensor[..., 0] - 1

    def f(optim_vars, aux_vars):
        (x,) = optim_vars
        mu, z = aux_vars
        augment = torch.sqrt(mu.tensor) * g(x.tensor) + z.tensor / (
            2.0 * torch.sqrt(mu.tensor)
        )
        cost = torch.cat([x.tensor, augment], axis=-1)
        return cost

    n_dim = 2
    n_constraints = z.shape[-1]
    x = th.Vector(n_dim, name="x")
    # mu = th.Variable(torch.ones(1, 1) * 1000, name="mu")
    # z = th.Variable(torch.ones(1, 1) * 1, name="z")
    optim_vars = [x]
    aux_vars = [mu, z]
    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        f,
        n_dim + n_constraints,
        aux_vars=aux_vars,
        name="cost_fn",
    )
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.GaussNewton(
        objective,
        max_iterations=15,
        step_size=1.0,
    )

    theseus_inputs = {
        "x": torch.ones([1, n_dim]).requires_grad_(),
    }
    theseus_optim = th.TheseusLayer(optimizer)
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
    )

    # print(updated_inputs)  # , info)
    return updated_inputs["x"]


mu = th.Variable(torch.ones(1, 1), name="mu")
z = th.Variable(torch.zeros(1, 2), name="z")

prev_g_norm = 0
# print(x)
for iter in range(10):
    x = aug_lagrang(mu, z)
    g_x = g(x)
    z.tensor = z.tensor + 2 * mu.tensor * g_x
    g_norm = g_x.norm()
    if g_norm > 0.25 * prev_g_norm:
        mu.tensor = 2 * mu.tensor
    prev_g_norm = g_norm
    print(x, mu, z)
