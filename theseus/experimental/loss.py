# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Tuple

import torch


class Loss(abc.ABC):
    @abc.abstractmethod
    def evaluate_scalar(self, s: torch.Tensor) -> torch.Tensor:
        pass

    def evaluate(self, error: torch.Tensor) -> torch.Tensor:
        error_squared_norm = torch.sum(error**2, dim=1, keepdim=True)
        return self.evaluate_scalar(error_squared_norm)

    @abc.abstractmethod
    def rescale(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass


class TrivialLoss(Loss):
    @abc.abstractmethod
    def evaluate_scalar(self, s: torch.Tensor) -> torch.Tensor:
        return s

    def rescale(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return jacobians, error


class RobustLoss(Loss):
    def __init__(self, log_loss_radius: torch.Tensor) -> None:
        self.log_loss_radius = log_loss_radius

    @abc.abstractmethod
    def evaluate_scalar(self, s: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def linearize(self, s: torch.Tensor) -> torch.Tensor:
        pass

    def rescale(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error_squared_norm = torch.sum(error**2, dim=1, keepdim=True)
        rescale = self.linearize(error_squared_norm).sqrt()

        return [
            (rescale * jacobian.view(jacobian.shape[0], -1)).view(jacobian.shape)
            for jacobian in jacobians
        ], rescale * error


class HuberLoss(RobustLoss):
    def __init__(self, log_loss_radius: torch.Tensor) -> None:
        super().__init__(log_loss_radius)

    def evaluate_scalar(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.exp()
        return torch.where(
            s > radius, 2 * torch.sqrt(radius * s.max(radius)) - radius, s
        )

    def linearize(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.exp()
        return torch.sqrt(radius / torch.max(s, radius))


class WelschLoss(RobustLoss):
    def __init__(self, log_loss_radius: torch.Tensor) -> None:
        super().__init__(
            log_loss_radius,
        )

    def evaluate_scalar(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.exp()
        return radius - radius * torch.exp(-s / radius)

    def linearize(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.exp()
        return torch.exp(-s / radius)
