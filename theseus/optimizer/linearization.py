# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional

from theseus.core import Objective

from .variable_ordering import VariableOrdering


class Linearization(abc.ABC):
    # if ordering is None, this will generate a default ordering
    def __init__(
        self,
        objective: Objective,
        ordering: Optional[VariableOrdering] = None,
        **kwargs
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
    def _linearize_hessian_impl(self):
        pass

    def linearize(self):
        if not self.ordering.complete:
            raise RuntimeError(
                "Attempted to linearize an objective with an incomplete variable order."
            )
        self._linearize_hessian_impl()

    def hessian_approx(self):
        raise NotImplementedError
