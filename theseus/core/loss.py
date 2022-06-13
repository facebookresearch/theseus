# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch

_LOSS_EPS = 1e-20


class RobustLoss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def evaluate(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def linearize(x: torch.Tensor, raidus: torch.Tensor) -> torch.Tensor:
        pass


class WelschLoss(RobustLoss):
    @staticmethod
    def evaluate(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return radius - radius * torch.exp(-x / (radius + _LOSS_EPS))

    @staticmethod
    def linearize(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x / (radius + _LOSS_EPS))


class HuberLoss(RobustLoss):
    @staticmethod
    def evaluate(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x > radius, 2 * torch.sqrt(radius * x.max(radius) + _LOSS_EPS) - radius, x
        )

    @staticmethod
    def linearize(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(radius / torch.max(x, radius) + _LOSS_EPS)
