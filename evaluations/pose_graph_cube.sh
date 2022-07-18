#!/bin/bash

# This script can be used to generate the results in Figure 3
# Options that can be changed for values used in the paper:
#  solver_device: {cuda, cpu}
#  batch_size: {1, 2, 4, 8, 16, 32, 64, 128, 256}
#  num_poses: {128, 256, 512, 1024, 2048, 4096}
python examples/pose_graph/pose_graph_cube.py \
       solver_device=cpu \
       device=cpu \
       dataset_size=256 \
       num_poses=256
