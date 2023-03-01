import torch

import theseus.labs.lie as lie

g1 = lie.rand(1, lie.SE3, requires_grad=True)
g2 = lie.rand(1, lie.SE3)

opt = torch.optim.Adam([g1], lr=0.1)

for i in range(10):
    opt.zero_grad()
    d = g2 - g1  # same as g1.local(g2)
    loss = torch.sum(d**2)
    loss.backward()
    opt.step()
    print(f"Iter {i}. Loss: {loss.item(): .3f}")
