#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This example illustrates the four backward modes
# (unroll, implicit, truncated, and dlm)
# on a problem fitting a quadratic to data.

import os

import torch

import theseus as th
from lie.functional import SE3 as SE3_Func
from theseus.labs.embodied.robot.forward_kinematics import (
    Robot,
    get_forward_kinematics_fns,
)

dtype = torch.float64
URDF_REL_PATH = (
    "../../tests/theseus_tests/embodied/kinematics/data/panda_no_gripper.urdf"
)
urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)

robot = Robot.from_urdf_file(urdf_path, dtype)
selected_links = ["panda_virtual_ee_link"]
fk, jfk_b, jfk_s = get_forward_kinematics_fns(robot, selected_links)


def compute_delta_theta(jfk, theta, targeted_pose, use_body_jacobian):
    # If jfk is body jacobian, delta_theta is computed by
    #          pose * exp(jfk * delta_theta) = targeted_pose,
    # which has a closed-form solution:
    # .  delta_theta = jfk.pinverse() * log(pose^-1 * targeted_pose)
    # Otherwise, if jfk is spatial jacobian, delta_theta is computed by
    #          exp(jfk * delta_theta) * pose = targeted_pose
    jac, poses = jfk(theta)
    pose_inv = SE3_Func.inv(poses[-1])
    error = (
        SE3_Func.log(
            SE3_Func.compose(pose_inv, targeted_pose)
            if use_body_jacobian
            else SE3_Func.compose(targeted_pose, pose_inv)
        )
        .view(-1, 6, 1)
        .view(-1, 6, 1)
    )
    return (jac[-1].pinverse() @ error).view(-1, robot.dof), error.norm().item()


print("---------------------------------------------------")
print("Body Jacobian")
print("---------------------------------------------------")
targeted_theta = torch.rand(100, robot.dof, dtype=dtype)
targeted_pose: torch.Tensor = fk(targeted_theta)[0]
theta_opt = torch.zeros_like(targeted_theta)
for iter in range(50):
    delta_theta, error = compute_delta_theta(jfk_b, theta_opt, targeted_pose, True)
    print(error)
    if error < 1e-4:
        break
    theta_opt = theta_opt + 0.5 * delta_theta

print("---------------------------------------------------")
print("Spatial Jacobian")
print("---------------------------------------------------")
targeted_theta = torch.rand(10, robot.dof, dtype=dtype)
targeted_pose = fk(targeted_theta)[0]
theta_opt = torch.zeros_like(targeted_theta)
for iter in range(50):
    delta_theta, error = compute_delta_theta(jfk_s, theta_opt, targeted_pose, False)
    print(error)
    if error < 1e-4:
        break
    theta_opt = theta_opt + 0.2 * delta_theta


print("---------------------------------------------------")
print("Theseus Optimizer")
print("---------------------------------------------------")


def targeted_pose_error(optim_vars, aux_vars):
    (theta,) = optim_vars
    (targeted_pose,) = aux_vars
    pose = th.SE3(tensor=fk(theta.tensor)[0])
    return (pose.inverse().compose(targeted_pose)).log_map()


optim_vars = (th.Vector(tensor=torch.zeros_like(theta_opt), name="theta_opt"),)
aux_vars = (th.SE3(tensor=targeted_pose, name="targeted_pose"),)


cost_function = th.AutoDiffCostFunction(
    optim_vars,
    targeted_pose_error,
    6,
    aux_vars=aux_vars,
    name="targeted_pose_error",
    cost_weight=th.ScaleCostWeight(torch.ones([1], dtype=dtype)),
    autograd_mode="vmap",
)
objective = th.Objective(dtype=dtype)
objective.add(cost_function)
optimizer = th.LevenbergMarquardt(
    objective,
    max_iterations=15,
    step_size=0.5,
    vectorize=True,
)

inputs = {"theta_opt": torch.zeros_like(theta_opt), "targeted_pose": targeted_pose}
optimizer.objective.update(inputs)
optimizer.optimize(verbose=True)