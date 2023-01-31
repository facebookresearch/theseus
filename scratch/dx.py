#!/usr/bin/env python3

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

x = torch.Tensor([0., 0. ,0.])
N = 10
us = torch.ones(N,2)

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
    x = f(x, us[i])
    xs.append(x.clone())
    print(x)



fig, ax = plt.subplots(figsize=(6,6))
for x, u in zip(xs, us):
    p = x[:2]
    theta = x[2]
    # print(p, theta)

    # ax.scatter(p[0], p[1], color='k')

    width = 0.05
    rect = patches.Rectangle(
         (0, -0.5*width), L, width, linewidth=1,
        edgecolor='black', facecolor='grey', alpha=0.5)
    t = mpl.transforms.Affine2D().rotate(theta).translate(*p) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    wheel_length = width/2.
    wheel = patches.Rectangle(
         (L-0.5*wheel_length, -0.5*width), wheel_length, 0, linewidth=1,
        edgecolor='black', alpha=1)
    t = mpl.transforms.Affine2D().rotate_around(
        L, -0.5*width, u[1]).rotate(theta).translate(*p) + ax.transData
    wheel.set_transform(t)
    ax.add_patch(wheel)

    wheel = patches.Rectangle(
         (L-0.5*wheel_length, +0.5*width), wheel_length, 0, linewidth=1,
        edgecolor='black', alpha=1)
    t = mpl.transforms.Affine2D().rotate_around(
        L, +0.5*width, u[1]).rotate(theta).translate(*p) + ax.transData
    wheel.set_transform(t)
    ax.add_patch(wheel)

# ax.set_xlim(-.5, .5)
# ax.set_ylim(0, 1)
ax.axis('equal')
fig.savefig('t.png')
