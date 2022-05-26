#!/bin/bash

# This script can be used to generate the results in Figure 2
# Options that can be changed for values used in the paper:
#  inner_optim.solver: {dense, sparse}
#  solver_device: {cuda, cpu}
#  batch_size: {8, 16, 32, 64, 128, 256}
#  num_poses: {128, 256, 512, 1024, 2048, 4096}

python examples/pose_graph_synthetic.py --multirun \
       hydra/launcher=submitit_slurm \
       hydra.launcher.partition=learnlab \
       hydra.launcher.timeout_min=4320 \
       hydra.launcher.cpus_per_task=20 \
       hydra.launcher.gpus_per_node=1 \
       hydra.launcher.mem_gb=128 \
       hydra.launcher.constraint=volta32gb \
       inner_optim.solver=sparse \
       loop_closure_ratio=0.2 \
       solver_device=cuda \
       dataset_size=256 \
       batch_size=128 \
       num_poses=256
