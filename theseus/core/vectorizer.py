from typing import List, Optional, Tuple

import torch

from .cost_function import CostFunction, _register_vars_in_list
from .objective import Objective

_CostFunctionSchema = Tuple[str]


def _get_cost_function_schema(cost_function: CostFunction) -> _CostFunctionSchema:
    def _fullname(obj):
        return f"{obj.__module__}.{obj.__class__.__name__}"

    def _varinfo(var):
        return f"{_fullname(var)}{tuple(var.shape[1:])}"

    return (
        (_fullname(cost_function),)
        + tuple(_varinfo(v) for v in cost_function.optim_vars)
        + tuple(_varinfo(v) for v in cost_function.aux_vars)
        + (_fullname(cost_function.weight),)
        + tuple(_varinfo(v) for v in cost_function.weight.aux_vars)
    )


# This wrapper allows the weighted error and jacobians to be replaced by pre-computed
# values. The vectorization class will compute these in a batch for all cost function,
# then populate wrappers for each so they can be served to the linearization classes.
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


# This class replaces the Objective's iterator for one that takes advantage of
# cost function vectorization
# TODO:
#   - Actually add vectorization logic
#   - Add some hook to call after Objective.update()
#   - Need to add code to clear error cache right at the beginning of every forward call
class Vectorize:
    def __init__(self, objective: Objective):
        self._cost_fn_wrappers: List[_CostFunctionWrapper] = []
        self._schema_dict: Dict[
            _CostFunctionSchema, List[_CostFunctionWrapper]
        ] = defaultdict(list)

        # Create wrappers for all cost functions and also get their schema
        for cost_function in objective.cost_functions.values():
            self._cost_fn_wrappers.append(_CostFunctionWrapper(cost_function))
            self._schema_dict[_get_cost_function_schema(cost_function)].append(
                cost_function
            )

        # Now create a vectorized cost function for each unique schema
        self._vectorized_cost_functions: Dict[_CostFunctionSchema, CostFunction] = dict(
            (schema, cost_fns[0].copy(keep_variable_names=True))
            for schema, cost_fns in self._schema_dict.items()
        )

        objective._cost_functions_iterable = self._cost_fn_wrappers


    