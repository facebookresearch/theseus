#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch

import theseus as th

torch.manual_seed(0)

def f(optim_vars, aux_vars):
    (x,) = optim_vars
    return x.tensor

n_dim = 2
x = th.Vector(n_dim, name="x")
optim_vars = [x]
cost_function = th.AutoDiffCostFunction(
    optim_vars,
    f,
    n_dim,
    name="cost_fn",
)
objective = th.Objective()
objective.add(cost_function)
optimizer = th.GaussNewton(
    objective,
    max_iterations=15,
    step_size=1.0,
)

theseus_inputs = {
    "x": torch.ones([1, n_dim]).requires_grad_(),
}
theseus_optim = th.TheseusLayer(optimizer)
updated_inputs, info = theseus_optim.forward(
    theseus_inputs,
)

print(updated_inputs) #, info)
