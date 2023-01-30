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

N = 10
L = 0.1
x_init = torch.zeros(3)
x_final = torch.tensor([0, 0.5, 0])
h = 0.1


def g(us, xs):
    constraints = [dx(x_init, us[0][0]) - xs[0]]
    for idx in range(N - 2):
        constraints.append(dx(xs[idx][0], us[idx][0]) - xs[idx + 1])
    constraints.append((dx(xs[-1][0], us[-1][0]) - x_final).unsqueeze(0))
    return torch.cat(constraints).ravel()


def dx(x, u):
    # new_x = x.clone()
    new_x = [
        x[0] + (h * u[0] * torch.cos(x[2])),
        x[1] + h * u[0] * torch.sin(x[2]),
        x[2] + h * u[0] * torch.tan(u[1] / L),
    ]
    return torch.stack(new_x)


def aug_lagrang(mu, z):
    # def g(optim_vars):
    #     (x,) = optim_vars
    #     return x.tensor[..., 0] - 1

    def f(optim_vars, aux_vars, lam=torch.tensor(0.1)):
        us, xs = optim_vars[:N], optim_vars[N:]
        us = [u.tensor for u in us]
        xs = [x.tensor for x in xs]
        cost_list = us
        for idx in range(len(us) - 1):
            cost_list.append(torch.sqrt(lam) * (us[idx + 1] - us[idx]))

        mu, z = aux_vars
        augment = torch.sqrt(mu.tensor) * g(us, xs) + z.tensor / (
            2.0 * torch.sqrt(mu.tensor)
        )
        # cost = torch.cat([x.tensor, augment], axis=-1)
        cost = torch.cat(cost_list + [augment], axis=-1)
        # print(cost.shape)
        return cost

    n_dim = N * 2
    n_constraints = z.shape[-1]
    u = [th.Vector(2, name="u" + str(idx + 1)) for idx in range(N)]
    x = [th.Vector(3, name="x" + str(idx + 1)) for idx in range(N - 1)]
    # mu = th.Variable(torch.ones(1, 1) * 1000, name="mu")
    # z = th.Variable(torch.ones(1, 1) * 1, name="z")
    optim_vars = u + x
    aux_vars = [mu, z]
    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        f,
        2 * (N + N - 1) + n_constraints,
        aux_vars=aux_vars,
        name="cost_fn",
    )
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=15,
        step_size=1.0,
    )

    # theseus_inputs = {
    #     "x": torch.ones([1, n_dim]).requires_grad_(),
    # }
    # theseus_inputs = {"u" + str(idx + 1): torch.ones([1, 2]) * 10 for idx in range(N)}
    theseus_optim = th.TheseusLayer(optimizer)
    # TODO: warmstart with previous solution
    updated_inputs, info = theseus_optim.forward(
        # theseus_inputs, optimizer_kwargs={"verbose": True}
    )

    # print(updated_inputs)  # , info)
    return updated_inputs


mu = th.Variable(torch.ones(1, 1), name="mu")
z = th.Variable(torch.zeros(1, N * 3), name="z")

prev_g_norm = 0
# print(x)
for iter in range(10):
    output = aug_lagrang(mu, z)
    us = [output["u" + str(idx + 1)] for idx in range(N)]
    xs = [output["x" + str(idx + 1)] for idx in range(N - 1)]
    # us = [u.tensor for u in us]
    # xs = [x.tensor for x in xs]

    g_x = g(us, xs)
    z.tensor = z.tensor + 2 * mu.tensor * g_x
    g_norm = g_x.norm()
    if g_norm > 0.25 * prev_g_norm:
        mu.tensor = 2 * mu.tensor
    prev_g_norm = g_norm

print(xs)
print(us)
