# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Type

import torch

from theseus.geometry.manifold import Manifold

from .cost_function import CostFunction
from .objective import Objective
from .variable import Variable

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
        self.register_vars(cost_fn.optim_vars, is_optim_vars=True)
        self.register_vars(cost_fn.aux_vars, is_optim_vars=False)
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
class Vectorize:
    _SHARED_TOKEN = "__shared__"

    def __init__(self, objective: Objective, empty_cuda_cache: bool = False):
        # Each cost function is assigned a wrapper. The wrapper will hold the error
        # and jacobian after vectorization.
        self._cost_fn_wrappers: List[_CostFunctionWrapper] = []

        # This dict groups cost functions that can be vectorized together.
        self._schema_dict: Dict[
            _CostFunctionSchema, List[_CostFunctionWrapper]
        ] = defaultdict(list)

        # Create wrappers for all cost functions and also get their schemas
        for cost_fn in objective.cost_functions.values():
            wrapper = _CostFunctionWrapper(cost_fn)
            self._cost_fn_wrappers.append(wrapper)
            schema = _get_cost_function_schema(cost_fn)
            self._schema_dict[schema].append(wrapper)

        # Now create a vectorized cost function for each unique schema
        self._vectorized_cost_fns: Dict[_CostFunctionSchema, CostFunction] = {}
        for schema in self._schema_dict:
            base_cost_fn = self._schema_dict[schema][0].cost_fn
            vectorized_cost_fn = base_cost_fn.copy(keep_variable_names=False)
            vectorized_cost_fn.weight = base_cost_fn.weight.copy(
                keep_variable_names=False
            )
            self._vectorized_cost_fns[schema] = vectorized_cost_fn

        # Dict[_CostFunctionSchema, List[str]]
        self._var_names = self._get_var_names()

        # `self._vectorize()` will compute an error vector for each schema,
        # then populate the wrappers with their appropriate weighted error slice.
        # Replacing `obj._cost_functions_iterable` allows to recover these when
        # iterating the Objective.
        objective._enable_vectorization(
            self._cost_fn_wrappers,
            self._vectorize,
            self._to,
            self._vectorized_retract_optim_vars,
            self,
        )

        self._objective = objective
        self._empty_cuda_cache = empty_cuda_cache

    # Returns a dictionary which maps every schema to information about its shared
    # variables.
    # The information is in the form of a list of strings for all variables in the
    # schema, such that if the variable is shared the string has the name of the
    # variable and the prefix __SHARED__. Otherwise it returns "var_{idx}",
    # where idx is the index of the var in the schema order.
    def _get_var_names(self) -> Dict[_CostFunctionSchema, List[Optional[str]]]:
        info = {}
        for schema, cost_fn_wrappers in self._schema_dict.items():
            # schema is (_, N optim_vars, M aux_vars, _, P aux_vars)
            # so the total number of vars is len(schema) - 2
            # var_names_per_position holds all variable names associated to each
            # position in schema. Shared variables are those positions with only one
            # name
            var_names_per_position: List[Set[str]] = [
                set() for _ in range(len(schema) - 2)
            ]
            for wrapper in cost_fn_wrappers:
                cf = wrapper.cost_fn
                var_idx = 0
                for var_iterators in [cf.optim_vars, cf.aux_vars, cf.weight.aux_vars]:
                    for v in var_iterators:
                        var_names_per_position[var_idx].add(v.name)
                        var_idx += 1
                assert var_idx == len(schema) - 2

            def _get_name(var_idx, name_set_):
                if len(name_set_) == 1:
                    return f"{Vectorize._SHARED_TOKEN}{next(iter(name_set_))}"
                return f"var_{var_idx}"

            info[schema] = [
                _get_name(i, s) for i, s in enumerate(var_names_per_position)
            ]
        return info

    @staticmethod
    def _get_all_vars(cf: CostFunction) -> List[Variable]:
        return (
            list(cf.optim_vars)
            + list(cf.aux_vars)  # type:ignore
            + list(cf.weight.aux_vars)  # type:ignore
        )

    @staticmethod
    def _expand(tensor: torch.Tensor, size: int) -> torch.Tensor:
        return tensor.expand((size,) + ((-1,) * (len(tensor.shape) - 1)))

    # Populates names_to_tensors with a list with one element per vectorized variable,
    # and which holds the concatenated tensors of its corresponding variables in all
    # cost functions in the list (which are assumed to have the same schema).
    # Inputs:
    #   - vars_names; The names for all variables in the cost_fns' schema. Shared
    #       variables must be prefixed with `_SHARED_TOKEN`.`
    #   - batch_size: the Objectve's batch size. Whenever a cost function's var tensor
    #       is batch_size 1, it expands it to this value, unless the variable is
    #       shared by all cost functions.
    #   - names_to_tensors: A dictionary mapping variable names to the list of tensors.
    #       It will be modified in place.
    @staticmethod
    def _update_all_cost_fns_var_tensors(
        cf_wrappers: List[_CostFunctionWrapper],
        var_names: List[str],
        objective_batch_size: int,
        names_to_tensors: Dict[str, List[torch.Tensor]],
    ):
        # Get all the tensors from individual variables
        for wrapper in cf_wrappers:
            cost_fn = wrapper.cost_fn
            cost_fn_vars = Vectorize._get_all_vars(cost_fn)
            # some shared vars may appear in more than one position in the list
            # For example GPMotionModel and GPCostWeight both have dt in their
            # list of vars. With this set we can skip repetitions
            seen_vars = set()
            for var_idx, var in enumerate(cost_fn_vars):
                name = var_names[var_idx]
                if name in seen_vars:
                    continue

                # If the variable is shared and batch_size == 1, only need a tensor for
                # one of the cost functions and we can just extend later to complete
                # the vectorized batch, so here we can do `continue`
                var_batch_size = var.tensor.shape[0]
                if (
                    name in names_to_tensors
                    and Vectorize._SHARED_TOKEN in name
                    and var_batch_size == 1
                ):
                    continue

                # Otherwise, we need to append to the list of tensors. since
                # we cannot extend to a full batch w/o copying.

                # If not a shared variable, expand to batch size if needed, because
                # those we will copy, no matter what.
                # For shared variables, just append, we will handle the expansion
                # when updating the vectorized variable containers.
                tensor = (
                    var.tensor
                    if (
                        var_batch_size > 1
                        or objective_batch_size == 1
                        or Vectorize._SHARED_TOKEN in name
                    )
                    else Vectorize._expand(var.tensor, objective_batch_size)
                )
                names_to_tensors[name].append(tensor)
                seen_vars.add(name)

    # Goes through the list of vectorized variables and updates them with the
    # concatenation of all tensors in their corresponding entry in
    # `names_to_tensors`. Shared variables are expanded to shape
    # batch_size * num_cost_fns
    @staticmethod
    def _update_vectorized_vars(
        all_vectorized_vars: List[Variable],
        names_to_tensors: Dict[str, List[torch.Tensor]],
        var_names: List[str],
        batch_size: int,
        num_cost_fns: int,
    ):
        for var_idx, var in enumerate(all_vectorized_vars):
            name = var_names[var_idx]

            if num_cost_fns == 1:
                var.update(names_to_tensors[name][0])
                continue

            all_var_tensors = names_to_tensors[name]
            if Vectorize._SHARED_TOKEN in name and all_var_tensors[0].shape[0] == 1:
                # In this case this is a shared variable, so all_var_tensors[i] is
                # the same for any value of i. So, we can just expand to the full
                # vectorized size. Sadly, this doesn't work if batch_size > 1
                var_tensor = Vectorize._expand(
                    all_var_tensors[0], batch_size * num_cost_fns
                )
            else:
                var_tensor = torch.cat(all_var_tensors, dim=0)
            var.update(var_tensor)

    # Computes the error of the vectorized cost function and distributes the error
    # to the cost function wrappers. The list of wrappers must correspond to the
    # same schema from which the vectorized cost function was obtained.
    @staticmethod
    def _compute_error_and_replace_wrapper_caches(
        vectorized_cost_fn: CostFunction,
        cost_fn_wrappers: List[_CostFunctionWrapper],
        batch_size: int,
    ):
        v_jac, v_err = vectorized_cost_fn.weighted_jacobians_error()
        start_idx = 0
        for wrapper in cost_fn_wrappers:
            assert wrapper._cached_error is None
            assert wrapper._cached_jacobians is None
            v_slice = slice(start_idx, start_idx + batch_size)
            wrapper._cached_error = v_err[v_slice]
            wrapper._cached_jacobians = [jac[v_slice] for jac in v_jac]
            start_idx += batch_size

    def _clear_wrapper_caches(self):
        for cost_fn_wrappers in self._schema_dict.values():
            for cf in cost_fn_wrappers:
                cf._cached_error = None
                cf._cached_jacobians = None
        if self._empty_cuda_cache:
            torch.cuda.empty_cache()

    # This could be a static method, but writing like this makes some unit tests
    # easier
    def _handle_singleton_wrapper(
        self, _: _CostFunctionSchema, cost_fn_wrappers: List[_CostFunctionWrapper]
    ):
        wrapper = cost_fn_wrappers[0]
        (
            wrapper._cached_jacobians,
            wrapper._cached_error,
        ) = wrapper.cost_fn.weighted_jacobians_error()

    def _handle_schema_vectorization(
        self, schema: _CostFunctionSchema, cost_fn_wrappers: List[_CostFunctionWrapper]
    ):
        var_names = self._var_names[schema]
        vectorized_cost_fn = self._vectorized_cost_fns[schema]
        all_vectorized_vars = Vectorize._get_all_vars(vectorized_cost_fn)
        assert len(all_vectorized_vars) == len(var_names)
        names_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
        batch_size = self._objective.batch_size

        Vectorize._update_all_cost_fns_var_tensors(
            cost_fn_wrappers, var_names, batch_size, names_to_tensors
        )
        Vectorize._update_vectorized_vars(
            all_vectorized_vars,
            names_to_tensors,
            var_names,
            batch_size,
            len(cost_fn_wrappers),
        )
        Vectorize._compute_error_and_replace_wrapper_caches(
            vectorized_cost_fn, cost_fn_wrappers, batch_size
        )

    def _vectorize(self):
        self._clear_wrapper_caches()
        for schema, cost_fn_wrappers in self._schema_dict.items():
            if len(cost_fn_wrappers) == 1:
                self._handle_singleton_wrapper(schema, cost_fn_wrappers)
            else:
                self._handle_schema_vectorization(schema, cost_fn_wrappers)

    @staticmethod
    def _vectorized_retract_optim_vars(
        delta: torch.Tensor,
        ordering: Iterable[Manifold],
        ignore_mask: Optional[torch.Tensor] = None,
        force_update: bool = False,
    ):
        # Each (variable-type, dof) gets mapped to a tuple with:
        #   - the variable that will hold the vectorized tensor
        #   - all the variables of that type that will be vectorized together
        #   - the delta slices that go into the batch
        var_info: Dict[
            Tuple[Type[Manifold], int],
            Tuple[Manifold, List[Manifold], List[torch.Tensor]],
        ] = {}

        start_idx = 0
        batch_size = -1
        # Create var info by looping variables in the given order
        # All variables of the same type get grouped together, and their deltas saved
        for var in ordering:
            if batch_size == -1:
                batch_size = var.shape[0]
            else:
                assert batch_size == var.shape[0]

            delta_slice = delta[:, start_idx : start_idx + var.dof()]
            start_idx += var.dof()
            var_type = (var.__class__, var.dof())
            if var_type not in var_info:
                var_info[var_type] = (var.copy(), [], [])
            var_info[var_type][1].append(var)
            var_info[var_type][2].append(delta_slice)

        for _, (vectorized_var, var_list, delta_list) in var_info.items():
            n_vars, dof = len(var_list), vectorized_var.dof()

            # Get the vectorized tensor that holds all the current variable tensors.
            # The resulting shape is (N * b, M), b is batch size, N is the number of
            # variables in the group, and M is the data shape for this class
            vectorized_tensor = torch.cat([v.tensor for v in var_list], dim=0)
            assert (
                vectorized_tensor.shape
                == (n_vars * batch_size,) + vectorized_tensor.shape[1:]
            )
            # delta_list is a list of N tensors of shape (b, d), where
            # d is var.dof(). After vectorization, we get tensor of shape (N * b, d)
            vectorized_delta = torch.cat(delta_list, dim=0)
            assert vectorized_delta.shape == (n_vars * batch_size, dof)

            # Retract the vectorized tensor, then redistribute to the variables
            vectorized_var.update(vectorized_tensor)
            new_var = vectorized_var.retract(vectorized_delta)
            start_idx = 0
            for var in var_list:
                retracted_tensor_slice = new_var[start_idx : start_idx + batch_size, :]
                if ignore_mask is None or force_update:
                    var.update(retracted_tensor_slice)
                else:
                    var.update(retracted_tensor_slice, batch_ignore_mask=ignore_mask)
                start_idx += batch_size

    # Applies to() with given args to all vectorized cost functions in the objective
    def _to(self, *args, **kwargs):
        for cost_function in self._vectorized_cost_fns.values():
            cost_function.to(*args, **kwargs)
