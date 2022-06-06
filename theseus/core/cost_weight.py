# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional, Sequence, Tuple, Union, cast

import torch

from .theseus_function import TheseusFunction
from .variable import Variable


# Abstract class for representing cost weights (aka, precisions, inverse covariance)
# Concrete classes must implement two methods:
#   - `weight_error`: return an error tensor weighted by the cost weight
#   - `weightJacobiansError`: returns jacobians an errors weighted by the cost weight
class CostWeight(TheseusFunction, abc.ABC):
    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

    @abc.abstractmethod
    def weight_error(self, error: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def weight_jacobians_and_error(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass

    # Must copy everything
    @abc.abstractmethod
    def _copy_impl(self, new_name: Optional[str] = None) -> "TheseusFunction":
        pass

    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "CostWeight":
        return cast(
            CostWeight,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )


# Besides passing a theseus Variable, can also get a float and it will create the
# Variable with a default name for it
class ScaleCostWeight(CostWeight):
    def __init__(
        self,
        scale: Union[float, torch.Tensor, Variable],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if not isinstance(scale, Variable):
            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale)
            self.scale = Variable(scale)
        else:
            self.scale = scale
        if not self.scale.data.squeeze().ndim in [0, 1]:
            raise ValueError("ScaleCostWeight only accepts 0- or 1-dim (batched) data.")
        self.scale.data = self.scale.data.view(-1, 1)
        self.register_aux_vars(["scale"])

    def weight_error(self, error: torch.Tensor) -> torch.Tensor:
        return error * self.scale.data

    def weight_jacobians_and_error(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error = error * self.scale.data
        new_jacobians = []
        for jac in jacobians:
            new_jacobians.append(jac * self.scale.data.view(-1, 1, 1))
        return new_jacobians, error

    def _copy_impl(self, new_name: Optional[str] = None) -> "ScaleCostWeight":
        return ScaleCostWeight(self.scale.copy(), name=new_name)


# Besides passing a theseus Variable, can also get any float sequence and it will create the
# Variable with a default name for it
class DiagonalCostWeight(CostWeight):
    def __init__(
        self,
        diagonal: Union[Sequence[float], Variable],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if not isinstance(diagonal, Variable):
            self.diagonal = Variable(torch.tensor(diagonal))
        else:
            self.diagonal = diagonal
        if not self.diagonal.data.squeeze().ndim in [1, 2]:
            raise ValueError("DiagonalCostWeight only accepts 1-D variables.")
        if self.diagonal.data.ndim == 1:
            self.diagonal.data = self.diagonal.data.unsqueeze(0)
        self.register_aux_vars(["diagonal"])

    def weight_error(self, error: torch.Tensor) -> torch.Tensor:
        return error * self.diagonal.data

    def weight_jacobians_and_error(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error = error * self.diagonal.data
        new_jacobians = []
        for jac in jacobians:
            # Jacobian is batch_size x cost_fuction_dim x var_dim
            # This left multiplies the weights to jacobian
            new_jacobians.append(jac * self.diagonal.data.unsqueeze(2))
        return new_jacobians, error

    def _copy_impl(self, new_name: Optional[str] = None) -> "DiagonalCostWeight":
        return DiagonalCostWeight(self.diagonal.copy(), name=new_name)
