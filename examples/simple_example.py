# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# A minimal example using Theseus that is fitting a curve to a dataset of observations.

import torch

import theseus as th


def y_model(x, c):
    return c * torch.exp(x)


def generate_data(num_points=10, c=0.5):
    data_x = torch.linspace(-1, 1, num_points).view(1, -1)
    data_y = y_model(data_x, c)
    return data_x, data_y


def read_data():
    num_points = 20
    data_x, data_y_clean = generate_data(num_points)
    return data_x, data_y_clean, 0.5 * torch.ones(1, 1)


x_true, y_true, v_true = read_data()  # shapes (1, N), (1, N), (1, 1)
x = th.Variable(torch.randn_like(x_true), name="x")
y = th.Variable(y_true, name="y")
v = th.Vector(1, name="v")  # a manifold subclass of Variable for optim_vars


def error_fn(optim_vars, aux_vars):  # returns y - v * exp(x)
    x, y = aux_vars
    return y.tensor - optim_vars[0].tensor * torch.exp(x.tensor)


objective = th.Objective()
cost_function = th.AutoDiffCostFunction(
    [v], error_fn, y_true.shape[1], aux_vars=[x, y], cost_weight=th.ScaleCostWeight(1.0)
)
objective.add(cost_function)
layer = th.TheseusLayer(th.GaussNewton(objective, max_iterations=10))

phi = torch.nn.Parameter(x_true + 0.1 * torch.ones_like(x_true))
outer_optimizer = torch.optim.Adam([phi], lr=0.001)
for epoch in range(20):
    solution, info = layer.forward(
        input_tensors={"x": phi.clone(), "v": torch.ones(1, 1)},
        optimizer_kwargs={"backward_mode": "implicit"},
    )
    outer_loss = torch.nn.functional.mse_loss(solution["v"], v_true)
    outer_loss.backward()
    outer_optimizer.step()
    print("Outer loss: ", outer_loss.item())
