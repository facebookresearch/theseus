# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Sequence

from theseus.core import Objective, Variable


class VariableOrdering:
    def __init__(self, objective: Objective, default_order: bool = True):
        self.objective = objective
        self._var_order: List[Variable] = []
        self._var_name_to_index: Dict[str, int] = {}
        if default_order:
            self._compute_default_order(objective)

    def _compute_default_order(self, objective: Objective):
        assert not self._var_order and not self._var_name_to_index
        cur_idx = 0
        for variable_name, variable in objective.optim_vars.items():
            if variable_name in self._var_name_to_index:
                continue
            self._var_order.append(variable)
            self._var_name_to_index[variable_name] = cur_idx
            cur_idx += 1

    def index_of(self, key: str) -> int:
        return self._var_name_to_index[key]

    def __getitem__(self, index) -> Variable:
        return self._var_order[index]

    def __iter__(self):
        return iter(self._var_order)

    def append(self, var: Variable):
        if var in self._var_order:
            raise ValueError(
                f"Variable {var.name} has already been added to the order."
            )
        if var.name not in self.objective.optim_vars:
            raise ValueError(
                f"Variable {var.name} is not an optimization variable for the objective."
            )
        self._var_order.append(var)
        self._var_name_to_index[var.name] = len(self._var_order) - 1

    def remove(self, var: Variable):
        self._var_order.remove(var)
        del self._var_name_to_index[var.name]

    def extend(self, variables: Sequence[Variable]):
        for var in variables:
            self.append(var)

    @property
    def complete(self):
        return len(self._var_order) == self.objective.size_variables()
