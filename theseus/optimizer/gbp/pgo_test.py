# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import theseus as th
from theseus.optimizer.gbp import GaussianBeliefPropagation, GBPSchedule

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


def create_pgo():

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

    gt_poses = []

    p = 0
    for i in range(size):
        for j in range(size):
            init = torch.Tensor([j, i])
            gt_poses.append(init[None, :])
            if i == 0 and j == 0:
                w = anchor_w
            else:
                # noise_init = torch.normal(torch.zeros(2), prior_noise_std)
                init = init + torch.FloatTensor(init_noises[p])
                w = prior_w

            prior_target = th.Vector(data=init, name=f"prior_{p}")
            inputs[f"x{p}"] = init[None, :]
            inputs[f"prior_{p}"] = init[None, :]

            cf_prior = th.Difference(poses[p], w, prior_target, name=f"prior_cost_{p}")

            objective.add(cf_prior)

            p += 1

    # Measurement cost functions

    meas_std_tensor = torch.nn.Parameter(torch.tensor([0.1]))
    meas_w = th.ScaleCostWeight(1 / meas_std_tensor, name="prior_weight")

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

    return objective, gt_poses, meas_std_tensor, inputs


def linear_solve_pgo():
    print("\n\nLinear solver...\n")

    objective, gt_poses, meas_std_tensor, inputs = create_pgo()

    # outer optimizer
    gt_poses_tensor = torch.cat(gt_poses)
    lr = 1e-3
    outer_optimizer = torch.optim.Adam([meas_std_tensor], lr=lr)
    outer_optimizer.zero_grad()

    linear_optimizer = th.LinearOptimizer(objective, th.CholeskyDenseSolver)
    th_layer = th.TheseusLayer(linear_optimizer)
    outputs_linsolve, _ = th_layer.forward(inputs, {"verbose": True})

    out_ls_tensor = torch.cat(list(outputs_linsolve.values()))
    loss = torch.norm(gt_poses_tensor - out_ls_tensor)
    loss.backward()

    print("loss", loss.item())
    print("grad", meas_std_tensor.grad.item())

    print("outputs\n", outputs_linsolve)


def gbp_solve_pgo(backward_mode, max_iterations=20):
    print("\n\nWith GBP...")
    print("backward mode:", backward_mode, "\n")

    objective, gt_poses, meas_std_tensor, inputs = create_pgo()

    gt_poses_tensor = torch.cat(gt_poses)
    lr = 1e-3
    outer_optimizer = torch.optim.Adam([meas_std_tensor], lr=lr)
    outer_optimizer.zero_grad()

    vectorize = True

    optimizer = GaussianBeliefPropagation(
        objective,
        max_iterations=max_iterations,
        vectorize=vectorize,
    )
    theseus_optim = th.TheseusLayer(optimizer, vectorize=vectorize)

    optim_arg = {
        "verbose": True,
        # "track_best_solution": True,
        # "track_err_history": True,
        "backward_mode": backward_mode,
        "backward_num_iterations": 5,
        "relin_threshold": 1e-8,
        "damping": 0.0,
        "dropout": 0.0,
        "schedule": GBPSchedule.SYNCHRONOUS,
    }

    outputs_gbp, info = theseus_optim.forward(inputs, optim_arg)

    out_gbp_tensor = torch.cat(list(outputs_gbp.values()))
    loss = torch.norm(gt_poses_tensor - out_gbp_tensor)
    loss.backward()

    print("loss", loss.item())
    print("grad", meas_std_tensor.grad.item())

    print("outputs\n", outputs_gbp)


linear_solve_pgo()

gbp_solve_pgo(backward_mode=th.BackwardMode.FULL, max_iterations=20)

gbp_solve_pgo(backward_mode=th.BackwardMode.TRUNCATED, max_iterations=20)
