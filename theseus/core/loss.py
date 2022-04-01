# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch


class Loss:
    @staticmethod
    @abc.abstractmethod
    def evaluate(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def linearize(x: torch.Tensor, raidus: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TrivialLoss(Loss):
    @staticmethod
    def evaluate(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def linearize(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], dtype=x.dtype)


class WelschLoss(Loss):
    @staticmethod
    def evaluate(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return radius - radius * torch.exp(-x / radius)

    @staticmethod
    def linearize(x: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        return torch.exp(-x / radius)
