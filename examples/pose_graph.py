# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import theseus as th

import theseus.utils.examples as theg

torch.manual_seed(1)

file_path = "datasets/tinyGrid3D.g2o"

num_verts, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path)

objective = th.Objective(torch.float64)

for edge in edges:
    cost_function = theg.RelativePoseError(
        verts[edge.i], verts[edge.j], edge.relative_pose, edge.weight
    )
    objective.add(cost_function)

torch.manual_seed(1)

file_path = "datasets/tinyGrid3D.g2o"

th.SO3.SO3_EPS = 1e-6

num_verts, verts, edges = theg.pose_graph.read_3D_g2o_file(file_path)

objective = th.Objective(torch.float64)

for edge in edges:
    cost_func = theg.RelativePoseError(
        verts[edge.i], verts[edge.j], edge.relative_pose, edge.weight
    )
    objective.add(cost_func)

objective.add(
    theg.PosePirorError(
        pose=verts[0], pose_prior=verts[0].copy(new_name=verts[0].name + "PRIOR")
    )
)

optimizer = th.LevenbergMarquardt(  # GaussNewton(
    objective,
    max_iterations=20,
    step_size=0.5,
)

theseus_optim = th.TheseusLayer(optimizer)

inputs = {var.name: var.data for var in verts}
theseus_optim.forward(inputs, optimizer_kwargs={"verbose": True})
