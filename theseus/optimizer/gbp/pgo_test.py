# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import theseus as th
from theseus.optimizer.gbp import GaussianBeliefPropagation

# This example illustrates the Gaussian Belief Propagation (GBP) optimizer
# for a 2D pose graph optimization problem.
# Linear problem where we are estimating the (x, y)position of 9 nodes,
# arranged in a 3x3 grid.
# Linear factors connect each node to its adjacent nodes.

np.random.seed(1)
torch.manual_seed(0)

size = 3
dim = 2

noise_cov = np.array([0.01, 0.01])

prior_noise_std = 0.2
prior_sigma = np.array([1.3**2, 1.3**2])

init_noises = np.random.normal(np.zeros([size * size, 2]), prior_noise_std)
meas_noises = np.random.normal(np.zeros([100, 2]), np.sqrt(noise_cov[0]))

# create theseus objective -------------------------------------

objective = th.Objective()
inputs = {}

n_poses = size * size

# create variables
poses = []
for i in range(n_poses):
    poses.append(th.Vector(data=torch.rand(1, 2), name=f"x{i}"))

# add prior cost constraints with VariableDifference cost
prior_std = 1.3
anchor_std = 0.01
prior_w = th.ScaleCostWeight(1 / prior_std, name="prior_weight")
anchor_w = th.ScaleCostWeight(1 / anchor_std, name="anchor_weight")

p = 0
for i in range(size):
    for j in range(size):
        init = torch.Tensor([j, i])
        if i == 0 and j == 0:
            w = anchor_w
        else:
            # noise_init = torch.normal(torch.zeros(2), prior_noise_std)
            init = init + torch.FloatTensor(init_noises[p])
            w = prior_w

        prior_target = th.Vector(data=init, name=f"prior_{p}")
        inputs[f"x{p}"] = init[None, :]
        inputs[f"prior_{p}"] = init[None, :]

        cf_prior = th.eb.VariableDifference(
            poses[p], w, prior_target, name=f"prior_cost_{p}"
        )

        objective.add(cf_prior)

        p += 1

# Measurement cost functions

meas_std = 0.1
meas_w = th.ScaleCostWeight(1 / meas_std, name="prior_weight")

m = 0
for i in range(size):
    for j in range(size):
        if j < size - 1:
            measurement = torch.Tensor([1.0, 0.0])
            # measurement += torch.normal(torch.zeros(2), meas_std)
            measurement += torch.FloatTensor(meas_noises[m])
            ix0 = i * size + j
            ix1 = i * size + j + 1

            meas = th.Vector(data=measurement, name=f"meas_{m}")
            inputs[f"meas_{m}"] = measurement[None, :]

            cf_meas = th.eb.Between(
                poses[ix0], poses[ix1], meas_w, meas, name=f"meas_cost_{m}"
            )
            objective.add(cf_meas)
            m += 1

        if i < size - 1:
            measurement = torch.Tensor([0.0, 1.0])
            # measurement += torch.normal(torch.zeros(2), meas_std)
            measurement += torch.FloatTensor(meas_noises[m])
            ix0 = i * size + j
            ix1 = (i + 1) * size + j

            meas = th.Vector(data=measurement, name=f"meas_{m}")
            inputs[f"meas_{m}"] = measurement[None, :]

            cf_meas = th.eb.Between(
                poses[ix0], poses[ix1], meas_w, meas, name=f"meas_cost_{m}"
            )
            objective.add(cf_meas)
            m += 1

# # objective.update(init_dict)
# print("Initial cost:", objective.error_squared_norm())

# fg.print(brief=True)

# # for vis ---------------

# joint = fg.get_joint()
# marg_covs = np.diag(joint.cov())[::2]
# map_soln = fg.MAP().reshape([size * size, 2])

# Solve with Gauss Newton ---------------

# print("inputs", inputs)

optimizer = GaussianBeliefPropagation(
    objective,
    max_iterations=100,
)
theseus_optim = th.TheseusLayer(optimizer)

optim_arg = {
    "track_best_solution": True,
    "track_err_history": True,
    "verbose": True,
    "backward_mode": th.BackwardMode.FULL,
    "damping": 0.6,
    "dropout": 0.0,
}
updated_inputs, info = theseus_optim.forward(inputs, optim_arg)

print("updated_inputs", updated_inputs)
print("info", info)


# optimizer = th.GaussNewton(
#     objective,
#     max_iterations=15,
#     step_size=0.5,
# )
# theseus_optim = th.TheseusLayer(optimizer)

# with torch.no_grad():
#     optim_args = {"track_best_solution": True, "verbose": True}
#     updated_inputs, info = theseus_optim.forward(inputs, optim_args)
# print("updated_inputs", updated_inputs)
# print("info", info)

# import ipdb; ipdb.set_trace()
