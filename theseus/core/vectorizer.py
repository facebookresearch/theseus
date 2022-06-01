from typing import List, Optional, Tuple

import torch

from .cost_function import CostFunction, _register_vars_in_list
from .objective import Objective


# This wrapper takes a weighted cost function and makes a new one whose auxiliary
# variables are the union of the cost function and the cost weight.
class _CostFunctionWrapper(CostFunction):
    def __init__(self, cost_fn: CostFunction):
        super().__init__(cost_fn.weight, name=cost_fn.name)
        self.cost_fn = cost_fn
        _register_vars_in_list(self, cost_fn.optim_vars, is_optim=True)
        aux_vars = [v for v in cost_fn.aux_vars]
        aux_vars.extend([v for v in cost_fn.weight.aux_vars if v not in aux_vars])
        _register_vars_in_list(self, aux_vars, is_optim=False)

    def error(self) -> torch.Tensor:
        return self.cost_fn.error()

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_fn.jacobians()

    def dim(self) -> int:
        return self.cost_fn.dim()

    def _copy_impl(self, new_name: Optional[str] = None) -> "CostFunction":
        return self.cost_fn.copy(new_name=new_name)


class Vectorizer:
    def __init__(self, objective: Objective):
        self._cost_function_wrappers: List[_CostFunctionWrapper] = []

        for cost_function in objective.cost_functions.values():
            self._cost_function_wrappers.append(_CostFunctionWrapper(cost_function))
