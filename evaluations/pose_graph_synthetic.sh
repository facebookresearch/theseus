#!/bin/bash

# This script can be used to generate the results in Figure 2
# Options that can be changed for values used in the paper:
#  inner_optim.solver: {dense, sparse}
#  solver_device: {cuda, cpu}
#  batch_size: {8, 16, 32, 64, 128, 256}
#  num_poses: {128, 256, 512, 1024, 2048, 4096}
#
# When using inner_optim.solver=sparse, this script supports the following options:
#     - solver_device=cuda, solver_type can be lucuda|baspacho
#     - solver_device=cpu, then solver is always CHOLMOD (solver_type is ignored)
python examples/pose_graph/pose_graph_synthetic.py \
       inner_optim.solver=sparse \
       loop_closure_ratio=0.2 \
       solver_device=cpu \
       dataset_size=256 \
       batch_size=128 \
       num_poses=256 \
       solver_type=lucuda
