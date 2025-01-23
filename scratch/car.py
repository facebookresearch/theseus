#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is the car example starting on page 23 of:
# http://www.seas.ucla.edu/~vandenbe/133B/lectures/nllseq.pdf

import numpy as np
import torch
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import theseus as th
from augmented_lagrangian import solve_augmented_lagrangian

plt.style.use("bmh")

N = 20  # Number of timesteps
L = 0.1  # Length of car
h = 0.05  # Discretization interval
squared_cost_weight = 1e-2  # Weight for the controls
slew_cost_weight = 1.0  # Weight for the slew rate

x_init = torch.zeros(3)  # Initial state
x_final = torch.tensor([0.5, 0.3, np.pi / 2.0])  # Final state

# Can try these other final states, but isn't very stable.
# x_final = torch.tensor([0, 0.5, 0])
# x_final = torch.tensor([0, 1., 0])
# x_final = torch.tensor([0.5, 0.3, 0])
# x_final = torch.tensor([0, 1., 0.5*np.pi])
# x_final = torch.tensor([0.5, 0.5, 0.])


# Given a state x and action u, return the
# next state following the discretized dynamics.
def car_dynamics(x, u):
    new_x = [
        x[0] + h * u[0] * torch.cos(x[2]),
        x[1] + h * u[0] * torch.sin(x[2]),
        x[2] + h * u[0] * torch.tan(u[1] / L),
    ]
    return torch.stack(new_x)


def plot_car_trajectory(xs, us):
    # The optimization formulation leaves out these known values from the states
    xs = [x_init] + xs + [x_final]
    us = us + [torch.zeros(2)]

    fig, ax = plt.subplots(figsize=(6, 6))
    for x, u in zip(xs, us):
        x = x.squeeze()
        u = u.squeeze()

        p = x[:2]
        theta = x[2]

        width = 0.5 * L
        alpha = 0.5
        rect = patches.Rectangle(
            (0, -0.5 * width),
            L,
            width,
            linewidth=1,
            edgecolor="black",
            facecolor="#DCDCDC",
            alpha=alpha,
        )
        t = mpl.transforms.Affine2D().rotate(theta).translate(*p) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        wheel_length = width / 2.0
        wheel = patches.Rectangle(
            (L - 0.5 * wheel_length, -0.5 * width),
            wheel_length,
            0,
            linewidth=1,
            edgecolor="black",
            alpha=alpha,
        )
        t = (
            mpl.transforms.Affine2D()
            .rotate_around(L, -0.5 * width, u[1])
            .rotate(theta)
            .translate(*p)
            + ax.transData
        )
        wheel.set_transform(t)
        ax.add_patch(wheel)

        wheel = patches.Rectangle(
            (L - 0.5 * wheel_length, +0.5 * width),
            wheel_length,
            0,
            linewidth=1,
            edgecolor="black",
            alpha=alpha,
        )
        t = (
            mpl.transforms.Affine2D()
            .rotate_around(L, +0.5 * width, u[1])
            .rotate(theta)
            .translate(*p)
            + ax.transData
        )
        wheel.set_transform(t)
        ax.add_patch(wheel)

    ax.axis("equal")
    fig.savefig("trajectory.png")
    plt.close(fig)


def plot_action_sequence(us):
    us = torch.stack(us).squeeze()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(us[:, 0], label="speed")
    ax.plot(us[:, 1], label="steering angle")
    fig.legend(ncol=2, loc="upper center")
    fig.savefig("actions.png")
    plt.close(fig)


# Ensure the trajectory satisfies the dynamics and reaches the goal state
def equality_constraint_fn(optim_vars, aux_vars):
    xs, us = optim_vars[: N - 1], optim_vars[N - 1 :]
    us = [u.tensor for u in us]
    xs = [x.tensor for x in xs]
    assert len(us) == N
    constraints = [car_dynamics(x_init, us[0][0]) - xs[0]]
    for idx in range(N - 2):
        constraints.append(car_dynamics(xs[idx][0], us[idx + 1][0]) - xs[idx + 1])
    constraints.append((car_dynamics(xs[-1][0], us[-1][0]) - x_final).unsqueeze(0))
    constraints = torch.cat(constraints).ravel()
    return constraints


# Make the controls and rate of change of controls minimal.
def err_fn(optim_vars, aux_vars):
    xs, us = optim_vars[: N - 1], optim_vars[N - 1 :]
    us = [u.tensor for u in us]
    xs = [x.tensor for x in xs]
    err_list = copy.copy([u * squared_cost_weight for u in us])
    for idx in range(len(us) - 1):
        err_list.append(slew_cost_weight * (us[idx + 1] - us[idx]))

    err = torch.cat(err_list, axis=-1)
    return err


u = [th.Vector(2, name="u" + str(idx + 1)) for idx in range(N)]
x = [th.Vector(3, name="x" + str(idx + 1)) for idx in range(1, N)]
optim_vars = x + u
dim = 2 * (N + N - 1)


def plot_callback(state_dict):
    xs = [state_dict["x" + str(idx + 1)] for idx in range(1, N)]
    us = [state_dict["u" + str(idx + 1)] for idx in range(N)]
    plot_car_trajectory(xs, us)
    plot_action_sequence(us)


solve_augmented_lagrangian(
    err_fn=err_fn,
    optim_vars=optim_vars,
    aux_vars=[],
    dim=dim,
    equality_constraint_fn=equality_constraint_fn,
    optimizer_cls=th.LevenbergMarquardt,
    optimizer_kwargs=dict(max_iterations=100, step_size=1.0),
    verbose=True,
    initial_state_dict={},
    callback=plot_callback,
)
