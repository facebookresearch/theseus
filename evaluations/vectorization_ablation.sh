#!/bin/bash

# This can be used to get the results in Figure 1
# You can vary:
#   num_poses to be one of {32,64,128,256,512,1024,2048}
#   batch_size to be one of {16,32,64,128,256}
#   inner_optim.vectorize in {true,false}, toggles whether vectorization
#      will be used or not.
python examples/pose_graph/pose_graph_synthetic.py \
    solver_device=cpu \
    outer_optim.num_epochs=200 \
    outer_optim.max_num_batches=1 \
    inner_optim.max_iters=1 \
    batch_size=32 \
    num_poses=128 \
    inner_optim.vectorize=true
