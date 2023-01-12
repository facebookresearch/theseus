# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import List, Optional, Sequence, Tuple, Union, cast

import torch

from .theseus_function import TheseusFunction
from .variable import Variable, as_variable


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

    # Returns boolean indicators for zero weights in the batch
    @abc.abstractmethod
    def is_zero(self) -> torch.Tensor:
        pass

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
        self.scale = as_variable(scale)
        if not self.scale.tensor.squeeze().ndim in [0, 1]:
            raise ValueError(
                "ScaleCostWeight only accepts 0- or 1-dim (batched) tensors."
            )
        self.scale.tensor = self.scale.tensor.view(-1, 1)
        self.register_aux_vars(["scale"])

    def is_zero(self) -> torch.Tensor:
        return self.scale.tensor.squeeze(1) == 0

    def weight_error(self, error: torch.Tensor) -> torch.Tensor:
        return error * self.scale.tensor

    def weight_jacobians_and_error(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error = error * self.scale.tensor
        new_jacobians = []
        for jac in jacobians:
            new_jacobians.append(jac * self.scale.tensor.view(-1, 1, 1))
        return new_jacobians, error

    def _copy_impl(self, new_name: Optional[str] = None) -> "ScaleCostWeight":
        return ScaleCostWeight(self.scale.copy(), name=new_name)


# Besides passing a theseus Variable, can also get any float sequence and it will create the
# Variable with a default name for it
class DiagonalCostWeight(CostWeight):
    def __init__(
        self,
        diagonal: Union[Sequence[float], torch.Tensor, Variable],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.diagonal = as_variable(diagonal)
        if not self.diagonal.tensor.squeeze().ndim < 3:
            raise ValueError("DiagonalCostWeight only accepts tensors with ndim < 3.")
        if self.diagonal.tensor.ndim == 0:
            self.diagonal.tensor = self.diagonal.tensor.view(1, 1)
        if self.diagonal.tensor.ndim == 1:
            warnings.warn(
                "1-D diagonal input is ambiguous. Dimension will be "
                "interpreted as dof dimension and not batch dimension."
            )
            self.diagonal.tensor = self.diagonal.tensor.view(1, -1)
        self.register_aux_vars(["diagonal"])

    def is_zero(self) -> torch.Tensor:
        # The minimum of each (diagonal[b] == 0) is True only if all its elements are 0
        return (self.diagonal.tensor == 0).min(dim=1)[0].bool()

    def weight_error(self, error: torch.Tensor) -> torch.Tensor:
        return error * self.diagonal.tensor

    def weight_jacobians_and_error(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error = error * self.diagonal.tensor
        new_jacobians = []
        for jac in jacobians:
            # Jacobian is batch_size x cost_fuction_dim x var_dim
            # This left multiplies the weights to jacobian
            new_jacobians.append(jac * self.diagonal.tensor.unsqueeze(2))
        return new_jacobians, error

    def _copy_impl(self, new_name: Optional[str] = None) -> "DiagonalCostWeight":
        return DiagonalCostWeight(self.diagonal.copy(), name=new_name)
