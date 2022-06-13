# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch

_LOSS_EPS = 1e-20


class RobustLoss(abc.ABC):
    @classmethod
    def evaluate(cls, x: torch.Tensor, log_radius: torch.Tensor) -> torch.Tensor:
        return cls._evaluate_impl(x, log_radius.exp())

    @classmethod
    def linearize(cls, x: torch.Tensor, log_radius: torch.Tensor) -> torch.Tensor:
        return cls._linearize_impl(x, log_radius.exp())

    @staticmethod
    @abc.abstractmethod
    def _evaluate_impl(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def _linearize_impl(x: torch.Tensor, raidus: torch.Tensor) -> torch.Tensor:
        pass


class WelschLoss(RobustLoss):
    @staticmethod
    def _evaluate_impl(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return radius - radius * torch.exp(-x / (radius + _LOSS_EPS))

    @staticmethod
    def _linearize_impl(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x / (radius + _LOSS_EPS))


class HuberLoss(RobustLoss):
    @staticmethod
    def _evaluate_impl(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x > radius, 2 * torch.sqrt(radius * x.max(radius) + _LOSS_EPS) - radius, x
        )

    @staticmethod
    def _linearize_impl(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(radius / torch.max(x, radius) + _LOSS_EPS)
