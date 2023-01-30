#!/usr/bin/env python3

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x = torch.Tensor([0., 0. ,0.])
N = 10
u = torch.ones(N,2)

L = 0.1

def f(x, u):
    h = 0.1
    new_x = x.clone()
    new_x[0] += h*u[0]*torch.cos(x[2])
    new_x[1] += h*u[0]*torch.sin(x[2])
    new_x[2] += h*u[0]*torch.tan(u[1] / L)
    return new_x

xs = [x.clone()]
for i in range(N):
    x = f(x, u[i])
    xs.append(x.clone())
    print(x)



fig, ax = plt.subplots(figsize=(6,6))
for x in xs:
    p = x[:2]
    theta = x[2]
    ax.scatter(p[0], p[1], color='k')
    width = 0.05
    rect = patches.Rectangle(
         p, L, .5*width, linewidth=1,
        facecolor='grey', alpha=0.5)
    t2 = mpl.transforms.Affine2D().rotate_around(p[0], p[1], theta) + ax.transData
    rect.set_transform(t2)
    ax.add_patch(rect)

    rect = patches.Rectangle(
         p, L, -.5*width, linewidth=1,
        facecolor='grey', alpha=0.5)
    t2 = mpl.transforms.Affine2D().rotate_around(p[0], p[1], theta) + ax.transData
    rect.set_transform(t2)
    ax.add_patch(rect)

# ax.set_xlim(-.5, .5)
# ax.set_ylim(0, 1)
ax.axis('equal')
fig.savefig('t.png')
