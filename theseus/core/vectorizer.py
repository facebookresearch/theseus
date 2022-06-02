from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from .cost_function import CostFunction, _register_vars_in_list
from .objective import Objective

_CostFunctionSchema = Tuple[str, ...]


def _get_cost_function_schema(cost_function: CostFunction) -> _CostFunctionSchema:
    def _fullname(obj) -> str:
        return f"{obj.__module__}.{obj.__class__.__name__}"

    def _varinfo(var) -> str:
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
    def __init__(self, cost_fn: CostFunction, name: Optional[str] = None):
        if name is None:
            name = f"wrapper({cost_fn.name})"
        super().__init__(cost_fn.weight, name=name)
        self.cost_fn = cost_fn
        _register_vars_in_list(self, cost_fn.optim_vars, is_optim=True)
        _register_vars_in_list(self, cost_fn.aux_vars, is_optim=False)
        self._cached_error: Optional[torch.Tensor] = None
        self._cached_jacobians: Optional[List[torch.Tensor]] = None

    def error(self) -> torch.Tensor:
        return self.cost_fn.error()

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.cost_fn.jacobians()

    def weighted_error(self) -> torch.Tensor:
        if self._cached_error is None:
            return super().weighted_error()
        return self._cached_error

    def weighted_jacobians_error(
        self,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if self._cached_jacobians is None:
            return super().weighted_jacobians_error()
        assert self._cached_error is not None
        return self._cached_jacobians, self._cached_error

    def dim(self) -> int:
        return self.cost_fn.dim()

    def _copy_impl(self, new_name: Optional[str] = None) -> "_CostFunctionWrapper":
        return _CostFunctionWrapper(self.cost_fn, name=new_name)

    def __str__(self) -> str:
        return f"Wrapper({str(_get_cost_function_schema(self.cost_fn))})"

    def __repr__(self) -> str:
        return str(self)


# This class replaces the Objective's iterator for one that takes advantage of
# cost function vectorization
# TODO:
#   - Actually add vectorization logic
#   - Add some hook to call after Objective.update()
#   - Need to add code to clear error cache right at the beginning of every forward call
#   - Tests to add:
#         + Test for shared_aux_var computation
#         + Test for schema computation
class Vectorize:
    def __init__(self, objective: Objective):
        self._cost_fn_wrappers: List[_CostFunctionWrapper] = []
        self._schema_dict: Dict[
            _CostFunctionSchema, List[_CostFunctionWrapper]
        ] = defaultdict(list)

        # Create wrappers for all cost functions and also get their schema
        for cost_function in objective.cost_functions.values():
            wrapper = _CostFunctionWrapper(cost_function)
            self._cost_fn_wrappers.append(wrapper)
            self._schema_dict[_get_cost_function_schema(cost_function)].append(wrapper)

        # Now create a vectorized cost function for each unique schema
        self._vectorized_cost_functions: Dict[_CostFunctionSchema, CostFunction] = dict(
            (schema, cost_fns[0].copy(keep_variable_names=True))
            for schema, cost_fns in self._schema_dict.items()
        )

        self._shared_vars_info = self._get_shared_vars_info()

        # `vectorize()` will compute an error vector for each schema, then populate
        # the wrappers with their appropriate weighted error slice.
        # Replacing `obj._cost_functions_iterable` allows to recover these when
        # iterating the Objective.
        objective._cost_functions_iterable = self._cost_fn_wrappers

    def _get_shared_vars_info(self) -> Dict[_CostFunctionSchema, List[bool]]:
        info = {}
        for schema, cost_fns in self._schema_dict.items():
            # schema is (_, N optim_vars, M aux_vars, _, P aux_vars)
            # so the total number of vars is len(schema) - 2
            # This list holds all variable names associated to each position in the
            # schema. Shared variables are those positions with only one name
            var_names_per_position: List[Set[str]] = [
                set() for _ in range(len(schema) - 2)
            ]
            for cf in cost_fns:
                var_idx = 0
                for var_iterators in [cf.optim_vars, cf.aux_vars, cf.weight.aux_vars]:
                    for v in var_iterators:
                        var_names_per_position[var_idx].add(v.name)
                        var_idx += 1
                assert var_idx == len(schema) - 2

            info[schema] = [len(name_set) == 1 for name_set in var_names_per_position]
        return info

