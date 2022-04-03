# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional, Tuple, cast

import torch

from theseus.core.theseus_function import TheseusFunction

from ..geometry import Vector


class LossFunction(TheseusFunction, abc.ABC):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    @abc.abstractmethod
    def function_value(self, error: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def rescale(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass

    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "LossFunction":
        return cast(
            LossFunction,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )


class TrivialLoss(LossFunction):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def function_value(self, error: torch.Tensor) -> torch.Tensor:
        return torch.sum(error**2, dim=1, keepdim=True)

    def rescale(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return jacobians, error

    def _copy_impl(self, new_name: Optional[str] = None) -> "TrivialLoss":
        return TrivialLoss(name=new_name)

    # only added to avoid casting downstream
    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "TrivialLoss":
        return cast(
            TrivialLoss,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )


class RobustLoss(LossFunction):
    def __init__(self, log_loss_radius: Vector, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.log_loss_radius = log_loss_radius
        self.register_aux_var("log_loss_radius")

    @abc.abstractmethod
    def evaluate(self, s: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def linearize(self, s: torch.Tensor) -> torch.Tensor:
        pass

    def function_value(self, error: torch.Tensor) -> torch.Tensor:
        error_squared_norm = torch.sum(error**2, dim=1, keepdim=True)
        return self.evaluate(error_squared_norm)

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

    # only added to avoid casting downstream
    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "RobustLoss":
        return cast(
            RobustLoss,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )


class HuberLoss(RobustLoss):
    def __init__(self, log_loss_radius: Vector, name: Optional[str] = None) -> None:
        super().__init__(log_loss_radius, name)

    def evaluate(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.data.exp()
        return torch.min(s, 2 * torch.sqrt(radius * s) - radius)

    def linearize(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.data.exp()
        return torch.sqrt(radius / torch.max(s, radius))

    def _copy_impl(self, new_name: Optional[str] = None) -> "HuberLoss":
        return HuberLoss(log_loss_radius=self.log_loss_radius.copy(), name=new_name)

    # only added to avoid casting downstream
    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "HuberLoss":
        return cast(
            HuberLoss,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )


class WelschLoss(RobustLoss):
    def __init__(self, log_loss_radius: Vector, name: Optional[str] = None) -> None:
        super().__init__(log_loss_radius, name)

    def evaluate(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.data.exp()
        return radius - radius * torch.exp(-s / radius)

    def linearize(self, s: torch.Tensor) -> torch.Tensor:
        radius = self.log_loss_radius.data.exp()
        return torch.exp(-s / radius)

    def _copy_impl(self, new_name: Optional[str] = None) -> "WelschLoss":
        return WelschLoss(log_loss_radius=self.log_loss_radius.copy(), name=new_name)

    # only added to avoid casting downstream
    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "WelschLoss":
        return cast(
            WelschLoss,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )
