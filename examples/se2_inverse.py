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


def lift(x: LieGroup):
    x.data.__class__ = torch.Tensor
    x.data.__dict__ = {}
    return x


def embed(x: LieGroup):
    x.data.__class__ = LieGroupTensor
    x.data.__dict__["group_cls"] = type(x)
    return x


def run(x1, x2, num_iters=10, embeded=True):
    if embeded:
        # update x1 with the Lie group tangent gradient
        x1 = embed(x1)
    else:
        # update x1 with the Euclidean gradient
        x1 = lift(x1)
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
print("Automatic Differentiation on the Lie Group Tangent Space")
print("---------------------------------------------------------")
run(x1.copy(), x2.copy(), num_iters=1000, embeded=True)

print("\n")
print("=========================================================")
print("Automatic Differentiation on the Euclidean Space")
print("---------------------------------------------------------")
run(x1.copy(), x2.copy(), num_iters=1000, embeded=False)
