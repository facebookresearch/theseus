#!/bin/bash

# This script can be used to generate the results in Figure 4
# Options that can be changed for values used in the paper:
#  inner_optim.backward_mode: {unroll, implicit, truncated, dlm}
#  inner_optim.max_iters: {2, 5, 10, 20, 30, 40, 50}
#
# When using DLM, set inner_optim.dlm_epsilon=0.01
# When using TRUNCATED, inner_optim.backward_num_iterations can be set to {5, 10} 
python examples/tactile_pose_estimation.py \
    train.num_epochs=100 \
    inner_optim.reg_w=0 \
    inner_optim.force_max_iters=true \
    inner_optim.force_implicit_by_epoch=10000 \
    train.lr=1e-4 \
    train.batch_size=8 \
    inner_optim.val_iters=50 \
    inner_optim.keep_step_size=true \
    inner_optim.step_size=0.05 \
    train.optimizer=adam \
    train.lr_decay=0.98 \
    inner_optim.max_iters=2 \
    inner_optim.dlm_epsilon=None \
    inner_optim.backward_num_iterations=None \
    inner_optim.backward_mode=implicit
