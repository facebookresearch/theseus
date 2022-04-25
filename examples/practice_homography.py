#! /usr/bin/env python3

import theseus as th
import torch


# A 3x3 homography transform.
H = th.Variable(torch.eye(3, 3), name="H")

# Source 2D point in homogeneous coords.
x = th.Variable(torch.Tensor([1,1,1]).reshape(3,1), name="x")

# Target 2D point in homogeneous coords.
xgt = th.Variable(torch.Tensor([2,2,1]).reshape(3,1), name="xgt")

def mse_error_fn(optim_vars, aux_vars):
    H = optim_vars
    x, xgt = aux_vars
    xp = H.data @ x.data
    xp.data = xp.data[:] / xp.data[2]
    err = (xgt.data - xp.data).square().mean()
    return err


# Set up cost function.
optim_vars = (H,)
aux_vars = (x, xgt)
cost_function = th.AutoDiffCostFunction(
    optim_vars, mse_error_fn, 100, aux_vars=aux_vars, name="mse_cost_fn"
)

# Set up objective.
objective = th.Objective()
objective.add(cost_function)

# Set up optimizer.
optimizer = th.GaussNewton(
    objective,
    max_iterations=15,
    step_size=0.5,
)
theseus_optim = th.TheseusLayer(optimizer)

# Run optimizer.
with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})

