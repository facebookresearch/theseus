# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from theseus.core import Objective

from .linearization import Linearization
from .variable_ordering import VariableOrdering


class DenseLinearization(Linearization):
    def __init__(
        self,
        objective: Objective,
        ordering: Optional[VariableOrdering] = None,
        **kwargs
    ):
        super().__init__(objective, ordering)
        self.A: torch.Tensor = None
        self.AtA: torch.Tensor = None
        self.b: torch.Tensor = None
        self.Atb: torch.Tensor = None

    def _linearize_jacobian_impl(self):
        err_row_idx = 0
        self.A = torch.zeros(
            (self.objective.batch_size, self.num_rows, self.num_cols),
            device=self.objective.device,
            dtype=self.objective.dtype,
        )
        self.b = torch.zeros(
            (self.objective.batch_size, self.num_rows),
            device=self.objective.device,
            dtype=self.objective.dtype,
        )
        for cost_function in self.objective._get_iterator():
            jacobians, error = cost_function.weighted_jacobians_error()
            num_rows = cost_function.dim()
            for var_idx_in_cost_function, var_jacobian in enumerate(jacobians):
                var_idx_in_order = self.ordering.index_of(
                    cost_function.optim_var_at(var_idx_in_cost_function).name
                )
                var_start_col = self.var_start_cols[var_idx_in_order]

                num_cols = var_jacobian.shape[2]
                row_slice = slice(err_row_idx, err_row_idx + num_rows)
                col_slice = slice(var_start_col, var_start_col + num_cols)
                self.A[:, row_slice, col_slice] = var_jacobian

            self.b[:, row_slice] = -error
            err_row_idx += cost_function.dim()

    def _linearize_hessian_impl(self):
        self._linearize_jacobian_impl()
        At = self.A.transpose(1, 2)
        self.AtA = At.bmm(self.A)
        self.Atb = At.bmm(self.b.unsqueeze(2))

    def hessian_approx(self):
        return self.AtA
