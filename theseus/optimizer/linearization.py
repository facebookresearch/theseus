# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional

import torch

from theseus.core import Objective

from .variable_ordering import VariableOrdering


class Linearization(abc.ABC):
    # if ordering is None, this will generate a default ordering
    def __init__(
        self,
        objective: Objective,
        ordering: Optional[VariableOrdering] = None,
        **kwargs,
    ):
        self.objective = objective
        if ordering is None:
            ordering = VariableOrdering(objective, default_order=True)
        self.ordering = ordering
        if not self.ordering.complete:
            raise ValueError("Given variable ordering is not complete.")

        self.var_dims: List[int] = []
        self.var_start_cols: List[int] = []
        col_counter = 0
        for var in ordering:
            v_dim = var.dof()
            self.var_start_cols.append(col_counter)
            self.var_dims.append(v_dim)
            col_counter += v_dim

        self.num_cols = col_counter
        self.num_rows = self.objective.dim()

    @abc.abstractmethod
    def _linearize_jacobian_impl(self):
        pass

    @abc.abstractmethod
    def _linearize_hessian_impl(self, _detach_hessian: bool = False):
        pass

    def linearize(self, _detach_hessian: bool = False):
        if not self.ordering.complete:
            raise RuntimeError(
                "Attempted to linearize an objective with an incomplete variable order."
            )
        self._linearize_hessian_impl(_detach_hessian=_detach_hessian)

    def hessian_approx(self):
        raise NotImplementedError(
            f"hessian_approx is not implemented for {self.__class__.__name__}"
        )

    @abc.abstractmethod
    def _ata_impl(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _atb_impl(self) -> torch.Tensor:
        pass

    @property
    def AtA(self) -> torch.Tensor:
        return self._ata_impl()

    @property
    def Atb(self) -> torch.Tensor:
        return self._atb_impl()

    # Returns self.A @ v
    @abc.abstractmethod
    def Av(self, v: torch.Tensor) -> torch.Tensor:
        pass

    # Returns diag(self.A^T @ self.A) * v
    @abc.abstractmethod
    def diagonal_scaling(self, v: torch.Tensor) -> torch.Tensor:
        pass
