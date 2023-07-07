#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# -------------------------------------------------------------------------- #
# ----------------- INVERSE KINEMATICS WITH THESEUS AND TORCHLIE ----------- #
# -------------------------------------------------------------------------- #
# In this example we show how to do inverse kinematics to achieve a target
# pose for a specific end effector of a robot described in a URDF file.
# We show two ways to accomplish this:
# 1) solve IK with body/spatial jacobian
# 2) solve IK as NLS optimization
import os

import torch

import theseus as th
from torchkin.forward_kinematics import Robot, get_forward_kinematics_fns
from torchlie.functional import SE3 as SE3_Func

dtype = torch.float64

# First we load the URDF file describing the robot and create a `Robot` object to
# represent it in Python. The `Robot` class can be used to build a kinematics tree
# of the robot.
URDF_REL_PATH = "../tests/theseus_tests/embodied/kinematics/data/panda_no_gripper.urdf"
urdf_path = os.path.join(os.path.dirname(__file__), URDF_REL_PATH)
robot = Robot.from_urdf_file(urdf_path, dtype)
link_names = ["panda_virtual_ee_link"]

# We can get differentiable forward kinematics functions for specific links
# by using `get_forward_kinematics_fns`. This function creates three differentiable
# functions for evaluating forward kinematics, body jacobian and spatial jacobian of
# the selected links, in that order. The return types of these functions are as
# follows:
#
# - fk: returns a tuple of link poses in the order of link names
# - jfk_b: returns a tuple where the first is a list of link body jacobians, and
#          the second is a tuple of link poses---both are in the order of link names
# - jfk_s: same as jfk_b except returning the spatial jacobians
fk, jfk_b, jfk_s = get_forward_kinematics_fns(robot, link_names)


# ********************************************************
# ** INVERSE KINEMATICS WITH Body/Spatial Jacobian
# ********************************************************
# If jfk is body jacobian, delta_theta is computed by
#          pose * exp(jfk * delta_theta) = targeted_pose,
# which has a closed-form solution:
# .  delta_theta = jfk.pinverse() * log(pose^-1 * targeted_pose)
# Otherwise, if jfk is spatial jacobian, delta_theta is computed by
#          exp(jfk * delta_theta) * pose = targeted_pose
def compute_delta_theta(jfk, theta, targeted_pose, use_body_jacobian):
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


# The following function runs IK. We first create random joint angles to compute
# a set of target poses for the end effector. Then, starting from zero joint angles,
# we try to recover joint angles that can achieve the target pose, by iteratively
# updating the joint angles using the body/spatial jacobian. Note that IK has infinite
# solutions, so, while we are able to recover the target poses after a few iterations,
# there is no guarantee that we can recover the joint angles used to generate these
# poses in the first place.
def run_ik_using_body_or_spatial_jacobian(
    jfk, batch_size, step_size, use_body_jacobian
):
    target_theta = torch.rand(batch_size, robot.dof, dtype=dtype)
    target_pose: torch.Tensor = fk(target_theta)[0]
    theta_opt = torch.zeros_like(target_theta)
    for _ in range(50):
        delta_theta, error = compute_delta_theta(
            jfk, theta_opt, target_pose, use_body_jacobian
        )
        print(error)
        if error < 1e-4:
            break
        theta_opt = theta_opt + step_size * delta_theta


print("---------------------------------------------------")
print("Use Body Jacobian")
print("---------------------------------------------------")
run_ik_using_body_or_spatial_jacobian(jfk_b, 100, 0.5, use_body_jacobian=True)

print("---------------------------------------------------")
print("Use Spatial Jacobian")
print("---------------------------------------------------")
run_ik_using_body_or_spatial_jacobian(jfk_s, 100, 0.2, use_body_jacobian=False)


# *********************************************
# ** INVERSE KINEMATICS AS NLS OPTIMIZATION
# *********************************************
print("---------------------------------------------------")
print("Theseus Optimizer")
print("---------------------------------------------------")


# IK can also be solved as an optimization problem:
#      min \|log(pose^-1 * targeted_pose)\|^2
# as the following
def targeted_pose_error(optim_vars, aux_vars):
    (theta,) = optim_vars
    (targeted_pose,) = aux_vars
    pose = th.SE3(tensor=fk(theta.tensor)[0])
    return pose.local(targeted_pose)


target_theta = torch.rand(10, robot.dof, dtype=dtype)
target_pose: torch.Tensor = fk(target_theta)[0]
theta_opt = torch.zeros_like(target_theta)
optim_vars = (th.Vector(tensor=torch.zeros_like(theta_opt), name="theta_opt"),)
aux_vars = (th.SE3(tensor=target_pose, name="targeted_pose"),)


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

inputs = {"theta_opt": torch.zeros_like(theta_opt), "targeted_pose": target_pose}
optimizer.objective.update(inputs)
optimizer.optimize(verbose=True)
