# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch


class RobustLoss:
    @staticmethod
    @abc.abstractmethod
    def evaluate(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def linearize(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TrivialLoss(RobustLoss):
    @staticmethod
    def evaluate(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return s

    @staticmethod
    def linearize(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(s)


class HuberLoss(RobustLoss):
    @staticmethod
    def evaluate(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.min(s, 2 * torch.sqrt(radius * s) - radius)

    @staticmethod
    def linearize(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(radius / torch.max(s, radius))


class WelschLoss(RobustLoss):
    @staticmethod
    def evaluate(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return radius - radius * torch.exp(-s / radius)

    @staticmethod
    def linearize(s: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.exp(-s / radius)
