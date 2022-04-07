# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import torch

from theseus.core.theseus_function import TheseusFunction
from theseus.geometry.manifold import Manifold

from .cost_function import CostFunction
from .cost_weight import CostWeight
from .variable import Variable


# If dtype is None, uses torch.get_default_dtype()
class Objective:
    def __init__(self, dtype: Optional[torch.dtype] = None):
        # maps variable names to the variable objects
        self.optim_vars: OrderedDict[str, Manifold] = OrderedDict()

        # maps variable names to variables objects, for optimization variables
        # that were registered when adding cost weights.
        self.cost_weight_optim_vars: OrderedDict[str, Manifold] = OrderedDict()

        # maps aux. variable names to the container objects
        self.aux_vars: OrderedDict[str, Variable] = OrderedDict()

        # maps variable name to variable, for any kind of variable added
        self._all_variables: OrderedDict[str, Variable] = OrderedDict()

        # maps cost function names to the cost function objects
        self.cost_functions: OrderedDict[str, CostFunction] = OrderedDict()

        # maps cost weights to the cost functions that use them
        # this is used when deleting cost function to check if the cost weight
        # variables can be deleted as well (when no other function uses them)
        self.cost_functions_for_weights: Dict[CostWeight, List[CostFunction]] = {}

        # ---- The following two methods are used just to get info from
        # ---- the objective, they don't affect the optimization logic.
        # a map from optimization variables to list of theseus functions it's
        # connected to
        self.functions_for_optim_vars: Dict[Manifold, List[TheseusFunction]] = {}

        # a map from all aux. variables to list of theseus functions it's connected to
        self.functions_for_aux_vars: Dict[Variable, List[TheseusFunction]] = {}

        self._batch_size: Optional[int] = None

        self.device: torch.device = torch.device("cpu")

        self.dtype: Optional[torch.dtype] = dtype or torch.get_default_dtype()

        # this increases after every add/erase operation, and it's used to avoid
        # an optimizer to run on a stale version of the objective (since changing the
        # objective structure might break optimizer initialization).
        self.current_version = 0

    def _add_function_variables(
        self,
        function: TheseusFunction,
        optim_vars: bool = True,
        is_cost_weight: bool = False,
    ):

        if optim_vars:
            function_vars = function.optim_vars
            self_var_to_fn_map = self.functions_for_optim_vars
            self_vars_of_this_type = (
                self.cost_weight_optim_vars if is_cost_weight else self.optim_vars
            )
        else:
            function_vars = function.aux_vars  # type: ignore
            self_var_to_fn_map = self.functions_for_aux_vars  # type: ignore
            self_vars_of_this_type = self.aux_vars  # type: ignore
        for variable in function_vars:
            # Check that variables have name and correct dtype
            if variable.name is None:
                raise ValueError(
                    f"Variables added to an objective must be named, but "
                    f"{function.name} has an unnamed variable."
                )
            if variable.dtype != self.dtype:
                raise ValueError(
                    f"Tried to add variable {variable.name} with data type "
                    f"{variable.dtype} but objective's data type is {self.dtype}."
                )
            # Check that names are unique
            if variable.name in self._all_variables:
                if variable is not self._all_variables[variable.name]:
                    raise ValueError(
                        f"Two different variable objects with the "
                        f"same name ({variable.name}) are not allowed "
                        "in the same objective."
                    )
            else:
                self._all_variables[variable.name] = variable
                assert variable not in self_var_to_fn_map
                self_var_to_fn_map[variable] = []

            # add to either self.optim_vars,
            # self.cost_weight_optim_vars or self.aux_vars
            self_vars_of_this_type[variable.name] = variable

            # add to list of functions connected to this variable
            self_var_to_fn_map[variable].append(function)

    # Adds a cost function to the objective
    # Also adds its optimization variables if they haven't been previously added
    # Throws an error if a new variable has the same name of a previously added
    # variable that is not the same object.
    # Does the same for the cost function's auxiliary variables
    # Then does the same with the cost weight's auxiliary variables
    #
    # For now, cost weight "optimization variables" are **NOT** added to the
    # set of objective's variables, they are kept in a separate container.
    # Update method will check if any of these are not registered as
    # cost function variables, and throw a warning.
    def add(self, cost_function: CostFunction):
        # adds the cost function if not already present
        if cost_function.name in self.cost_functions:
            if cost_function is not self.cost_functions[cost_function.name]:
                raise ValueError(
                    f"Two different cost function objects with the "
                    f"same name ({cost_function.name}) are not allowed "
                    "in the same objective."
                )
            else:
                warnings.warn(
                    "This cost function has already been added to the objective, "
                    "nothing to be done."
                )
        else:
            self.cost_functions[cost_function.name] = cost_function

        self.current_version += 1
        # ----- Book-keeping for the cost function ------- #
        # adds information about the optimization variables in this cost function
        self._add_function_variables(cost_function, optim_vars=True)

        # adds information about the auxiliary variables in this cost function
        self._add_function_variables(cost_function, optim_vars=False)

        if cost_function.weight not in self.cost_functions_for_weights:
            # ----- Book-keeping for the cost weight ------- #
            # adds information about the variables in this cost function's weight
            self._add_function_variables(
                cost_function.weight, optim_vars=True, is_cost_weight=True
            )
            # adds information about the auxiliary variables in this cost function's weight
            self._add_function_variables(
                cost_function.weight, optim_vars=False, is_cost_weight=True
            )

            self.cost_functions_for_weights[cost_function.weight] = []

            if cost_function.weight.num_optim_vars() > 0:
                warnings.warn(
                    f"The cost weight associated to {cost_function.name} receives one "
                    "or more optimization variables. Differentiating cost "
                    "weights with respect to optimization variables is not currently "
                    "supported, thus jacobians computed by our optimizers will be "
                    "incorrect. You may want to consider moving the weight computation "
                    "inside the cost function, so that the cost weight only receives "
                    "auxiliary variables.",
                    RuntimeWarning,
                )

        self.cost_functions_for_weights[cost_function.weight].append(cost_function)

        if self.optim_vars.keys() & self.aux_vars.keys():
            raise ValueError(
                "Objective does not support a variable being both "
                "an optimization variable and an auxiliary variable."
            )

    # returns a reference to the cost function with the given name
    def get_cost_function(self, name: str) -> CostFunction:
        return self.cost_functions.get(name, None)

    # checks if the cost function with the given name is in the objective
    def has_cost_function(self, name: str) -> bool:
        return name in self.cost_functions

    # checks if the optimization variable with the given name is in the objective
    def has_optim_var(self, name: str) -> bool:
        return name in self.optim_vars

    # returns a reference to the optimization variable with the given name
    def get_optim_var(self, name: str) -> Manifold:
        return self.optim_vars.get(name, None)

    # checks if the aux. variable with the given name is in the objective
    def has_aux_var(self, name: str) -> bool:
        return name in self.aux_vars

    # returns a reference to the aux. variable with the given name
    def get_aux_var(self, name: str) -> Variable:
        return self.aux_vars.get(name, None)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _erase_function_variables(
        self,
        function: TheseusFunction,
        optim_vars: bool = True,
        is_cost_weight: bool = False,
    ):
        if optim_vars:
            fn_var_list = function.optim_vars
            self_vars_of_this_type = (
                self.cost_weight_optim_vars if is_cost_weight else self.optim_vars
            )
            self_var_to_fn_map = self.functions_for_optim_vars
        else:
            fn_var_list = function.aux_vars  # type: ignore
            self_vars_of_this_type = self.aux_vars  # type: ignore
            self_var_to_fn_map = self.functions_for_aux_vars  # type: ignore

        for variable in fn_var_list:
            cost_fn_idx = self_var_to_fn_map[variable].index(function)
            # remove function from the variable's list of connected cost functions
            del self_var_to_fn_map[variable][cost_fn_idx]
            # if the variable has no other functions, remove it also
            if not self_var_to_fn_map[variable]:
                del self_var_to_fn_map[variable]
                del self_vars_of_this_type[variable.name]

    # Removes a cost function from the objective given its name
    # Also removes any of its variables that are no longer associated to other
    # functions (either cost functions, or cost weights).
    # Does the same for the cost weight, but only if the weight is not associated to
    # any other cost function
    def erase(self, name: str):
        self.current_version += 1
        if name in self.cost_functions:
            cost_function = self.cost_functions[name]
            # erase variables associated to this cost function (if needed)
            self._erase_function_variables(cost_function, optim_vars=True)
            self._erase_function_variables(cost_function, optim_vars=False)

            # delete cost function from list of cost functions connected to its weight
            cost_weight = cost_function.weight
            cost_fn_idx = self.cost_functions_for_weights[cost_weight].index(
                cost_function
            )
            del self.cost_functions_for_weights[cost_weight][cost_fn_idx]

            # No more cost functions associated to this weight, so can also delete
            if len(self.cost_functions_for_weights[cost_weight]) == 0:
                # erase its variables (if needed)
                self._erase_function_variables(
                    cost_weight, optim_vars=True, is_cost_weight=True
                )
                self._erase_function_variables(
                    cost_weight, optim_vars=False, is_cost_weight=True
                )
                del self.cost_functions_for_weights[cost_weight]

            # finally, delete the cost function
            del self.cost_functions[name]
        else:
            warnings.warn(
                "This cost function is not in the objective, nothing to be done."
            )

    # gets the name associated with a cost function object (or None if not present)
    def get_cost_function_name(self, cost_function: CostFunction) -> Optional[str]:
        for name in self.cost_functions:
            if id(cost_function) == id(cost_function):
                return name
        return None

    @staticmethod
    def _get_functions_connected_to_var(
        variable: Union[str, Variable],
        objectives_var_container_dict: "OrderedDict[str, Variable]",
        var_to_cost_fn_map: Dict[Variable, List[TheseusFunction]],
        variable_type: str,
    ) -> List[TheseusFunction]:
        if isinstance(variable, str):
            if variable not in objectives_var_container_dict:
                raise ValueError(
                    f"{variable_type} named {variable} is not in the objective."
                )
            variable = objectives_var_container_dict[variable]
        if variable not in var_to_cost_fn_map:
            raise ValueError(
                f"{variable_type} {variable.name} is not in the objective."
            )
        return var_to_cost_fn_map[variable]

    def get_functions_connected_to_optim_var(
        self, variable: Union[Manifold, str]
    ) -> List[TheseusFunction]:
        return Objective._get_functions_connected_to_var(
            variable,
            self.optim_vars,  # type: ignore
            self.functions_for_optim_vars,  # type: ignore
            "Optimization Variable",
        )

    def get_functions_connected_to_aux_var(
        self, aux_var: Union[Variable, str]
    ) -> List[TheseusFunction]:
        return Objective._get_functions_connected_to_var(
            aux_var, self.aux_vars, self.functions_for_aux_vars, "Auxiliary Variable"
        )

    # sum of cost function dimensions
    def dim(self) -> int:
        err_dim = 0
        for cost_function in self.cost_functions.values():
            err_dim += cost_function.dim()
        return err_dim

    # number of (cost functions, variables)
    def size(self) -> tuple:
        return len(self.cost_functions), len(self.optim_vars)

    # number of cost functions
    def size_cost_functions(self) -> int:
        return len(self.cost_functions)

    # number of variables
    def size_variables(self) -> int:
        return len(self.optim_vars)

    # number of auxiliary variables
    def size_aux_vars(self) -> int:
        return len(self.aux_vars)

    def error(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        also_update: bool = False,
    ) -> torch.Tensor:
        old_data = {}
        if input_data is not None:
            if not also_update:
                for var in self.optim_vars:
                    old_data[var] = self.optim_vars[var].data
            self.update(input_data=input_data)
        error_vector = torch.zeros(self.batch_size, len(self.cost_functions)).to(
            device=self.device, dtype=self.dtype
        )
        pos = 0
        for cost_function in self.cost_functions.values():
            error_vector[:, pos : pos + 1] = cost_function.value()
            pos += 1
        if not also_update:
            self.update(old_data)
        return error_vector

    def error_squared_norm(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        also_update: bool = False,
    ) -> torch.Tensor:
        return self.error(input_data=input_data, also_update=also_update).sum(dim=1)

    def copy(self) -> "Objective":
        new_objective = Objective()

        # First copy all individual cost weights
        old_to_new_cost_weight_map: Dict[CostWeight, CostWeight] = {}
        for cost_weight in self.cost_functions_for_weights:
            new_cost_weight = cost_weight.copy(
                new_name=cost_weight.name, keep_variable_names=True
            )
            old_to_new_cost_weight_map[cost_weight] = new_cost_weight

        # Now copy the cost functions and assign the corresponding cost weight copy
        new_cost_functions: List[CostFunction] = []
        for cost_function in self.cost_functions.values():
            new_cost_function = cost_function.copy(
                new_name=cost_function.name, keep_variable_names=True
            )
            # we assign the allocated weight copies to avoid saving duplicates
            new_cost_function.weight = old_to_new_cost_weight_map[cost_function.weight]
            new_cost_functions.append(new_cost_function)

        # Handle case where a variable is copied in 2+ cost functions or cost weights,
        # since only a single copy should be maintained by objective
        for cost_function in new_cost_functions:
            # CostFunction
            for i, var in enumerate(cost_function.optim_vars):
                if new_objective.has_optim_var(var.name):
                    cost_function.set_optim_var_at(
                        i, new_objective.optim_vars[var.name]
                    )
            for i, aux_var in enumerate(cost_function.aux_vars):
                if new_objective.has_aux_var(aux_var.name):
                    cost_function.set_aux_var_at(
                        i, new_objective.aux_vars[aux_var.name]
                    )
            # CostWeight
            for i, var in enumerate(cost_function.weight.optim_vars):
                if var.name in new_objective.cost_weight_optim_vars:
                    cost_function.weight.set_optim_var_at(
                        i, new_objective.cost_weight_optim_vars[var.name]
                    )
            for i, aux_var in enumerate(cost_function.weight.aux_vars):
                if new_objective.has_aux_var(aux_var.name):
                    cost_function.weight.set_aux_var_at(
                        i, new_objective.aux_vars[aux_var.name]
                    )
            new_objective.add(cost_function)
        return new_objective

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    def update(self, input_data: Optional[Dict[str, torch.Tensor]] = None):
        self._batch_size = None

        def _get_batch_size(batch_sizes: Sequence[int]) -> int:
            unique_batch_sizes = set(batch_sizes)
            if len(unique_batch_sizes) == 1:
                return batch_sizes[0]
            if len(unique_batch_sizes) == 2:
                min_bs = min(unique_batch_sizes)
                max_bs = max(unique_batch_sizes)
                if min_bs == 1:
                    return max_bs
            raise ValueError("Provided data tensors must be broadcastable.")

        input_data = input_data or {}
        for var_name, data in input_data.items():
            if data.ndim < 2:
                raise ValueError(
                    f"Input data tensors must have a batch dimension and "
                    f"one ore more data dimensions, but data.ndim={data.ndim} for "
                    f"tensor with name {var_name}."
                )
            if var_name in self.optim_vars:
                self.optim_vars[var_name].update(data)
            elif var_name in self.aux_vars:
                self.aux_vars[var_name].update(data)
            elif var_name in self.cost_weight_optim_vars:
                self.cost_weight_optim_vars[var_name].update(data)
                warnings.warn(
                    "Updated a variable declared as optimization, but it is "
                    "only associated to cost weights and not to any cost functions. "
                    "Theseus optimizers will only update optimization variables "
                    "that are associated to one or more cost functions."
                )
            else:
                warnings.warn(
                    f"Attempted to update a tensor with name {var_name}, "
                    "which is not associated to any variable in the objective."
                )

        # Check that the batch size of all data is consistent after update
        batch_sizes = [v.data.shape[0] for v in self.optim_vars.values()]
        batch_sizes.extend([v.data.shape[0] for v in self.aux_vars.values()])
        self._batch_size = _get_batch_size(batch_sizes)

    # iterates over cost functions
    def __iter__(self):
        return iter([f for f in self.cost_functions.values()])

    # Applies to() with given args to all tensors in the objective
    def to(self, *args, **kwargs):
        for cost_function in self.cost_functions.values():
            cost_function.to(*args, **kwargs)
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device or self.device
        self.dtype = dtype or self.dtype
