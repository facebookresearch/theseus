#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

import theseus as th
import theseus.utils.examples as theg

torch.set_default_dtype(torch.double)

device = "cuda:0" if torch.cuda.is_available else "cpu"
torch.random.manual_seed(1)
random.seed(1)
np.random.seed(1)

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["font.size"] = 16


# #### INITIAL SETUP
# Data and config stuff
dataset_dir = "tutorials/data/motion_planning_2d"
num_prob = 2
dataset = theg.TrajectoryDataset(True, num_prob, dataset_dir, "tarpit")
data_loader = torch.utils.data.DataLoader(dataset, num_prob, shuffle=False)

batch = next(iter(data_loader))
map_size = batch["map_tensor"].shape[1]
trajectory_len = batch["expert_trajectory"].shape[2]
num_time_steps = trajectory_len - 1
map_size = batch["map_tensor"].shape[1]
safety_distance = 1.5
robot_radius = 0.25
total_time = 10.0
dt_val = total_time / num_time_steps
Qc_inv = torch.eye(3)
collision_w = 20.0
boundary_w = 100.0

# Create the planner
planner = theg.MotionPlanner(
    optimizer_config=(
        "LevenbergMarquardt",
        {"max_optim_iters": 50, "step_size": 0.25},
    ),
    map_size=map_size,
    epsilon_dist=safety_distance + robot_radius,
    total_time=total_time,
    collision_weight=collision_w,
    Qc_inv=Qc_inv,
    num_time_steps=num_time_steps,
    device=device,
    pose_type=th.SE2,
    nonholonomic_w=10.0,
    positive_vel_w=5.0,
)

# #### INITIALIZE OPTIMIZER VARIABLES
start = torch.zeros(batch["expert_trajectory"].shape[0], 4)
start[:, :2] = batch["expert_trajectory"][:, :2, 0]
start[:, 3] = -1
goal = batch["expert_trajectory"][:, :2, -1]
planner_inputs = {
    "sdf_origin": batch["sdf_origin"].to(device),
    "start": start.to(device),
    "goal": goal.to(device),
    "cell_size": batch["cell_size"].to(device),
    "sdf_data": batch["sdf_data"].to(device),
}

# Initialize from straight line trajectory
initial_traj_dict = planner.get_variable_values_from_straight_line(
    planner_inputs["start"], planner_inputs["goal"]
)
planner_inputs.update(initial_traj_dict)

# #### RUN THE PLANNER
planner.layer.forward(
    planner_inputs,
    optimizer_kwargs={
        "verbose": True,
        "damping": 0.1,
    },
)

# #### PLOT SOLUTION
solution = planner.get_trajectory()

sdf = th.eb.SignedDistanceField2D(
    th.Point2(batch["sdf_origin"]),
    th.Variable(batch["cell_size"]),
    th.Variable(batch["sdf_data"]),
)
figures = theg.generate_trajectory_figs(
    batch["map_tensor"][1:].cpu(),
    sdf,
    [solution[1:].cpu()],
    robot_radius,
    max_num_figures=1,
    fig_idx_robot=0,
    labels=["solution"],
)
plt.show()
