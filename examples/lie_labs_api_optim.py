import torch

import theseus.labs.lie as lie
import theseus.labs.lie.functional as lieF

# Leaf tensor needs to be a regular tensor, so we need to explicitly pass
# the tensor data
g1_data = lieF.se3.rand(1, requires_grad=True)
g1 = lie.LieTensor.from_tensor(g1_data, lie.SE3)
g2 = lie.rand(1, lie.SE3)
opt = torch.optim.Adam([g1_data])

for i in range(10):
    opt.zero_grad()
    d = g1.inv().compose(g2).log()
    loss = torch.sum(d**2)
    loss.backward()
    opt.step()
    print(f"Iter {i}. Loss: {loss.item(): .3f}")
