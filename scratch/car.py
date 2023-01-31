#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import theseus as th
import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

N = 30
L = 0.05
x_init = torch.zeros(3)
# x_final = torch.tensor([0, 0.5, 0])
# x_final = torch.tensor([0, 1., 0])
# x_final = torch.tensor([0.5, 0.3, 0])
# x_final = torch.tensor([0, 1., 0.5*np.pi])
# x_final = torch.tensor([0.5, 0.5, np.pi/2.])
x_final = torch.tensor([0.5, 0.5, 0.])
h = 1.
gamma = 10.


def plot(xs, us):
    xs = [x_init] + xs + [x_final]
    us = us + [torch.zeros(2)]

    fig, ax = plt.subplots(figsize=(6,6))
    for x, u in zip(xs, us):
        x = x.squeeze()
        u = u.squeeze()

        p = x[:2]
        theta = x[2]

        width = 0.02
        alpha = 0.7
        rect = patches.Rectangle(
            (0, -0.5*width), L, width, linewidth=1,
            edgecolor='black', facecolor='#DCDCDC', alpha=alpha)
        t = mpl.transforms.Affine2D().rotate(theta).translate(*p) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        wheel_length = width/2.
        wheel = patches.Rectangle(
            (L-0.5*wheel_length, -0.5*width), wheel_length, 0, linewidth=1,
            edgecolor='black', alpha=alpha)
        t = mpl.transforms.Affine2D().rotate_around(
            L, -0.5*width, u[1]).rotate(theta).translate(*p) + ax.transData
        wheel.set_transform(t)
        ax.add_patch(wheel)

        wheel = patches.Rectangle(
            (L-0.5*wheel_length, +0.5*width), wheel_length, 0, linewidth=1,
            edgecolor='black', alpha=alpha)
        t = mpl.transforms.Affine2D().rotate_around(
            L, +0.5*width, u[1]).rotate(theta).translate(*p) + ax.transData
        wheel.set_transform(t)
        ax.add_patch(wheel)

    ax.axis('equal')
    fig.savefig('t.png')

def g(us, xs):
    constraints = [dx(x_init, us[0][0]) - xs[0]]
    for idx in range(N - 2):
        constraints.append(dx(xs[idx][0], us[idx][0]) - xs[idx + 1])
    constraints.append((dx(xs[-1][0], us[-1][0]) - x_final).unsqueeze(0))
    return torch.cat(constraints).ravel()


def dx(x, u):
    # new_x = x.clone()
    new_x = [
        x[0] + h * u[0] * torch.cos(x[2]),
        x[1] + h * u[0] * torch.sin(x[2]),
        x[2] + h * u[0] * torch.tan(u[1] / L),
    ]
    return torch.stack(new_x)


def aug_lagrang(mu, z, state_dict):
    def f(optim_vars, aux_vars):
        us, xs = optim_vars[:N], optim_vars[N:]
        us = [u.tensor for u in us]
        xs = [x.tensor for x in xs]
        cost_list = us
        for idx in range(len(us) - 1):
            cost_list.append(math.sqrt(gamma) * (us[idx + 1] - us[idx]))

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
    x = [th.Vector(3, name="x" + str(idx + 1)) for idx in range(1, N)]
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
    optimizer = th.GaussNewton(
        objective,
        max_iterations=10000,
        step_size=0.1,
    )

    theseus_optim = th.TheseusLayer(optimizer)
    with torch.no_grad():
        state_dict, info = theseus_optim.forward(
            state_dict,
            optimizer_kwargs={
                "verbose": True,
                # 'track_err_history': True,
                # 'track_state_history': True
            },
        )
    # assert all(info.status == th.NonlinearOptimizerStatus.CONVERGED )
    # import ipdb; ipdb.set_trace()

    # print(updated_inputs)  # , info)
    # import ipdb; ipdb.set_trace()
    return state_dict


mu = th.Variable(torch.ones(1, 1), name="mu")
z = th.Variable(torch.zeros(1, N * 3), name="z")

state_dict = {}

def project_states(state_dict):
    xt = x_init
    for t in range(1, N+1):
        # print(t)
        ut = state_dict.get(f'u{t}')
        # state_dict[f'u{t}'] = ut.unsqueeze(0)
        # print('h')
        state_dict[f'x{t+1}'] = dx(xt, ut.squeeze()).unsqueeze(0)
    # import ipdb; ipdb.set_trace()

prev_g_norm = 0

# increase_factor = 2.

# print(x)
for i in range(50):
    print(f'=== iter {i}')
    # if i > 0:
    #     project_states(state_dict)
    state_dict = aug_lagrang(mu, z, state_dict=state_dict)
    us = [state_dict["u" + str(idx + 1)] for idx in range(N)]
    xs = [state_dict["x" + str(idx + 1)] for idx in range(1, N)]
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
    print(mu)
    print(z)

    plot(xs, us)
    # import ipdb; ipdb.set_trace()

