import torch
import theseus as th
from theseus import LieGroupTensor

# BTW we should clean this function and make it accessible util under th.
from theseus.geometry.tests.test_se2 import create_random_se2

# This example will use torch to solve min_x1 || x1 - x2 ||^2,
# where x1 and x2 are SE2 objects
rng = torch.Generator()
rng.manual_seed(0)
x1 = create_random_se2(1, rng)  # bug: creates a random tensor w/o using rng
x2 = create_random_se2(1, rng)


def run(x1, x2, num_iters=10, use_proj=True):
    # LieGroupTensor defines how to update x1.data
    if use_proj:
        x1.data = LieGroupTensor(x1)
    x1.data.requires_grad = True
    optim = torch.optim.Adam([x1.data], lr=1e-1)
    for i in range(num_iters):
        optim.zero_grad()
        cf = th.eb.VariableDifference(x1.inverse(), th.ScaleCostWeight(1.0), x2)
        loss = cf.error().norm()
        loss.backward()
        optim.step()
        print(
            "loss: {:.10f},  cos(theta)^2 + sin(theta)^2 is {:.10f}".format(
                loss.item(), x1[0, 2:].norm().item()
            )
        )


print("===============================================")
print("Graident on the Lie Group Tangent Space")
print("-----------------------------------------------")
run(x1.copy(), x2.copy(), num_iters=10, use_proj=True)


print("===============================================")
print("Graident on the Euclidean Space")
print("-----------------------------------------------")
run(x1.copy(), x2.copy(), num_iters=10, use_proj=False)
