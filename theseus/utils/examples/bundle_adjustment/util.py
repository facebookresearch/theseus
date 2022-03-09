from typing import Tuple

import torch


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
