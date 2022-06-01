# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import torch


# ------------------------------------------------------------ #
# --------------------------- LOSSES ------------------------- #
# ------------------------------------------------------------ #
def soft_loss_cauchy(
    x: torch.Tensor, radius: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ratio = (x + radius) / radius
    val = torch.log(ratio) * radius
    der = 1.0 / ratio
    return val, der


def soft_loss_huber_like(
    x: torch.Tensor, radius: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ratio = (x + radius) / radius
    sq = torch.sqrt(ratio)
    val = (sq - 1) * radius
    der = 0.5 / sq
    return val, der


# For reprojection cost functions where the loss is 2 dimensional,
# x and y pixel loss, but the robust loss region is determined
# by the norm of the (x, y) pixel loss.
def soft_loss_huber_like_reprojection(
    x: torch.Tensor, radius: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_norm = torch.norm(x, dim=1).unsqueeze(1)
    val, der = soft_loss_huber_like(x_norm, radius)
    scaling = val / x_norm
    x_loss = x * scaling

    term1 = scaling[..., None] * torch.eye(2, dtype=x.dtype, device=x.device).reshape(
        1, 2, 2
    ).repeat(x.shape[0], 1, 1)
    term2 = (
        torch.bmm(x.unsqueeze(2), x.unsqueeze(1))
        * ((der - scaling) / (x_norm**2))[..., None]
    )
    der = term1 + term2

    return x_loss, der


# ------------------------------------------------------------ #
# ----------------------------- RNG -------------------------- #
# ------------------------------------------------------------ #
# returns a uniformly random point of the 2-sphere
def random_s2(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    theta = torch.rand(()) * math.tau
    z = torch.rand(()) * 2 - 1
    r = torch.sqrt(1 - z**2)
    return torch.tensor([r * torch.cos(theta), r * torch.sin(theta), z]).to(dtype=dtype)


# returns a uniformly random point of the 3-sphere
def random_s3(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    u, v, w = torch.rand(3)
    return torch.tensor(
        [
            torch.sqrt(1 - u) * torch.sin(math.tau * v),
            torch.sqrt(1 - u) * torch.cos(math.tau * v),
            torch.sqrt(u) * torch.sin(math.tau * w),
            torch.sqrt(u) * torch.cos(math.tau * w),
        ]
    ).to(dtype=dtype)


def random_small_quaternion(
    max_degrees: float, min_degrees: int = 0, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    x, y, z = random_s2(dtype=dtype)
    theta = (
        (min_degrees + (max_degrees - min_degrees) * torch.rand((), dtype=dtype))
        * math.tau
        / 360.0
    )
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.tensor([c, s * x, s * y, s * z])
