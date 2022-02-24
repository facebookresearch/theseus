#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This example will use torch to solve min_x1 || x1^-1 - x2 ||^2,
# where x1 and x2 are SE2 objects

import torch
import theseus as th
from theseus import LieGroupTensor
from theseus.geometry.lie_group import LieGroup

from theseus.geometry.tests.test_se2 import create_random_se2

rng = torch.Generator()
rng.manual_seed(0)
x1 = create_random_se2(1, rng)
x2 = create_random_se2(1, rng)


def run(x1: LieGroup, x2: LieGroup, num_iters=10, use_lie_tangent=True):
    x1.data = LieGroupTensor(x1)
    x1.data.requires_grad = True

    optim = torch.optim.Adam([x1.data], lr=1e-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[250, 600], gamma=0.01
    )
    for i in range(num_iters):
        optim.zero_grad()
        cf = th.eb.VariableDifference(x1.inverse(), th.ScaleCostWeight(1.0), x2)
        loss = cf.error().norm()
        if i % 100 == 0:
            print(
                "iter {:0>4d}: loss is {:.10f}, cos(theta)^2 + sin(theta)^2 is {:.10f}".format(
                    i, loss.item(), x1[0, 2:].norm().item() ** 2
                )
            )
        loss.backward()

        with th.set_lie_tangent_enabled(use_lie_tangent):
            optim.step()

        scheduler.step()

    cf = th.eb.VariableDifference(x1.inverse(), th.ScaleCostWeight(1.0), x2)
    loss = cf.error().norm()
    print(
        "iter {}: loss is {:.10f}, cos(theta)^2 + sin(theta)^2 is {:.10f}".format(
            num_iters, loss.item(), x1[0, 2:].norm().item() ** 2
        )
    )


print("=========================================================")
print("PyTorch Optimization on the Euclidean Space")
print("---------------------------------------------------------")
run(x1.copy(), x2.copy(), num_iters=1000, use_lie_tangent=False)
print("\n")

print("=========================================================")
print("PyTorch Optimization on the Lie Group Tangent Space (Ours)")
print("---------------------------------------------------------")
run(x1.copy(), x2.copy(), num_iters=1000, use_lie_tangent=True)
print("\n")
