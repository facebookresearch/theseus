# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import torch

from theseus.core.loss_function import LossFunction
from theseus.core.theseus_function import TheseusFunction
from theseus.core.variable import Variable
from theseus.geometry.manifold import Manifold

from .cost_function import CostFunction
from .cost_weight import CostWeight

# TODO: automatic batching of cost functions
# Assumptions:
# 1) Assume Objective.setup() must be called before running optimization
# 2) Assume Variable.update() must keep Variable.shape[1:]
# 3) Assume CostFunction.optim_vars and CostFunction.aux_vars must have the
#    same batch size as Objective.batch_size


# If dtype is None, uses torch.get_default_dtype()
class Objective:
    def __init__(self, dtype: Optional[torch.dtype] = None):
        # maps variable names to the variable objects
        self.optim_vars: OrderedDict[str, Manifold] = OrderedDict()

        # maps variable names to variables objects, for optimization variables
        # that were registered when adding cost weights.
        self.cost_weight_optim_vars: OrderedDict[str, Manifold] = OrderedDict()

        # maps variable names to variables objects, for optimization variables
        # that were registered when adding loss functions.
        self.loss_function_optim_vars: OrderedDict[str, Manifold] = OrderedDict()

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

        # maps loss functions to the cost functions that use them
        # this is used when deleting cost function to check if the loss function
        # variables can be deleted as well (when no other function uses them)
        self.cost_functions_for_loss_functions: Dict[
            LossFunction, List[CostFunction]
        ] = {}

        # maps cost function batch names to the corresponding batched cost functions
        self.batched_cost_functions: OrderedDict[
            str, Tuple[CostFunction, List[CostFunction]]
        ] = OrderedDict()

        # maps cost function names to unbatched functions
        self.unbatched_cost_functions: OrderedDict[str, CostFunction] = OrderedDict()

        # maps batched cost function names to their batch names
        self.batch_names_for_cost_functions: OrderedDict[str, str] = OrderedDict()

        # maps batched optim variable names to the the corresponding batched optim variables
        self.batched_optim_vars: OrderedDict[
            str, Tuple[Manifold, OrderedDict[str, Manifold]]
        ] = OrderedDict()

        # ---- The following two methods are used just to get info from
        # ---- the objective, they don't affect the optimization logic.
        # a map from optimization variables to list of theseus functions it's
        # connected to
        self.functions_for_optim_vars: Dict[Manifold, List[TheseusFunction]] = {}

        # a map from all aux. variables to list of theseus functions it's connected to
        self.functions_for_aux_vars: Dict[Variable, List[TheseusFunction]] = {}

        self._batch_size: Optional[int] = None

        self._is_setup: bool = False

        self.device: torch.device = torch.device("cpu")

        self.dtype: Optional[torch.dtype] = dtype or torch.get_default_dtype()

        # this increases after every add/erase operation, and it's used to avoid
        # an optimizer to run on a stale version of the objective (since changing the
        # objective structure might break optimizer initialization).
        self.current_version = 0

    @staticmethod
    def _get_cost_function_variable_batch_name(var: Variable) -> str:
        return (
            var.__module__
            + "."
            + var.__class__.__name__
            + "__"
            + f"{tuple(var.data.shape[1:])}"
        )

    @staticmethod
    def _get_cost_function_batch_name(cost_function: CostFunction) -> Union[str, None]:
        variables: List[Variable] = [
            optim_var for optim_var in cost_function.optim_vars
        ]
        variables.extend([aux_var for aux_var in cost_function.aux_vars])
        batch_sizes = [variable.shape[0] for variable in variables]
        unique_batch_sizes = set(batch_sizes)

        if len(unique_batch_sizes) != 1:
            return None

        batch_name = cost_function.__module__ + "." + cost_function.__class__.__name__

        variable_batch_names = [
            "-" + Objective._get_cost_function_variable_batch_name(var)
            for var in variables
        ]

        return batch_name + "".join(variable_batch_names)

    def _add_function_variables(
        self,
        function: TheseusFunction,
        optim_vars: bool = True,
        is_cost_weight: bool = False,
        is_loss_function: bool = False,
    ):

        if optim_vars:
            function_vars = function.optim_vars
            self_var_to_fn_map = self.functions_for_optim_vars
            if is_cost_weight:
                self_vars_of_this_type = self.cost_weight_optim_vars
            elif is_loss_function:
                self_vars_of_this_type = self.loss_function_optim_vars
            else:
                self_vars_of_this_type = self.optim_vars
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

                if self_vars_of_this_type is self.optim_vars:
                    batch_name = self._get_cost_function_variable_batch_name(variable)

                    if batch_name not in self.batched_optim_vars:
                        variable_batch = variable.copy(
                            new_name=batch_name + "__varialbe_batch"
                        )
                        optim_variables: OrderedDict[str, Manifold] = OrderedDict()
                        self.batched_optim_vars[batch_name] = (
                            variable_batch,
                            optim_variables,
                        )
                    else:
                        variable_batch, optim_variables = self.batched_optim_vars[
                            batch_name
                        ]

                        original_optim_variable = next(iter(optim_variables.values()))

                        if variable.shape != original_optim_variable.shape:
                            raise ValueError(
                                f"Tried to add variable {variable.name} with shape "
                                f"{tuple(variable.shape)} but batched variable's required  "
                                f"shape is {tuple(original_optim_variable.shape)}."
                            )

                    assert variable.name not in optim_variables
                    optim_variables[variable.name] = variable
                    variable_batch.data = None

            # add to either self.optim_vars,
            # self.cost_weight_optim_vars, self.loss_function_optim_vars or self.aux_vars
            self_vars_of_this_type[variable.name] = variable

            # add to list of functions connected to this variable
            self_var_to_fn_map[variable].append(function)

    def _reset_batched_cost_function_variables(
        self, cost_function: CostFunction, batch_name: str
    ):
        cost_function_batch, cost_functions = self.batched_cost_functions[batch_name]
        original_cost_function = cost_functions[0]

        if cost_function.__class__ != cost_function_batch.__class__:
            raise ValueError(
                f"Tried to add cost function {cost_function.name} with type "
                f"{type(cost_function)} but batched cost function's type is "
                f"{type(cost_function_batch)}."
            )

        vars_attr_names_list = [
            cost_function_batch._optim_vars_attr_names,
            cost_function_batch._aux_vars_attr_names,
        ]

        for vars_attr_names in vars_attr_names_list:
            for attr_name in vars_attr_names:
                variable = cast(Variable, getattr(cost_function, attr_name))
                original_variable = cast(
                    Variable, getattr(original_cost_function, attr_name)
                )

                if type(variable) != type(original_variable):
                    raise ValueError(
                        f"The type of variable {variable.name} in cost function "
                        f"{cost_function.name} is {type(variable)} and different "
                        f"from the expected type {type(original_variable)}."
                    )

                if variable.shape != original_variable.shape:
                    raise ValueError(
                        f"The shape of variable {variable.name} in cost function "
                        f"{cost_function.name} is {variable.shape} and different "
                        f"from the expected shape {original_variable.shape}."
                    )

                variable_batch = cast(Variable, getattr(cost_function_batch, attr_name))
                variable_batch.data = None

    def _add_batched_cost_function(
        self, cost_function: CostFunction, use_batches: bool = False
    ):
        batch_name = (
            self._get_cost_function_batch_name(cost_function) if use_batches else None
        )
        if batch_name is None:
            self.unbatched_cost_functions[cost_function.name] = cost_function
        else:
            if batch_name not in self.batched_cost_functions:
                cost_function_batch = cost_function.copy(
                    batch_name + "__cost_function_batch"
                )
                cost_function_batch.weight = None
                cost_function_batch.loss_function = None

                batch_sizes = []

                vars_attr_names_list = [
                    cost_function_batch._optim_vars_attr_names,
                    cost_function_batch._aux_vars_attr_names,
                ]
                for vars_attr_names in vars_attr_names_list:
                    for var_attr_name in vars_attr_names:
                        variable_batch = cast(
                            Variable, getattr(cost_function_batch, var_attr_name)
                        )
                        variable_batch.name = cost_function.name + "__" + var_attr_name
                        batch_sizes.append(variable_batch.data.shape[0])
                        variable_batch.data = None

                unique_batch_sizes = set(batch_sizes)

                if len(unique_batch_sizes) != 1:
                    raise ValueError("Provided cost function can not be batched.")

                self.batched_cost_functions[batch_name] = (
                    cost_function_batch,
                    [cost_function],
                )
            else:
                self._reset_batched_cost_function_variables(cost_function, batch_name)
                self.batched_cost_functions[batch_name][1].append(cost_function)

            self.batch_names_for_cost_functions[cost_function.name] = batch_name

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

    def add(self, cost_function: CostFunction, use_batches: bool = False):
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

        # adds information about cost function types

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

        if cost_function.loss_function not in self.cost_functions_for_loss_functions:
            # ----- Book-keeping for the loss function ------- #
            # adds information about the variables in this cost function's loss function
            self._add_function_variables(
                cost_function.loss_function, optim_vars=True, is_loss_function=True
            )
            # adds information about the auxiliary variables in this cost function's loss function
            self._add_function_variables(
                cost_function.loss_function, optim_vars=False, is_loss_function=True
            )

            self.cost_functions_for_loss_functions[cost_function.loss_function] = []

            if cost_function.loss_function.num_optim_vars() > 0:
                warnings.warn(
                    f"The loss function associated to {cost_function.name} receives one "
                    "or more optimization variables. Differentiating loss functions "
                    "with respect to optimization variables is not currently "
                    "supported, thus jacobians computed by our optimizers will be "
                    "incorrect. You may want to consider moving the loss function computation "
                    "inside the cost function, so that the loss function only receives "
                    "auxiliary variables.",
                    RuntimeWarning,
                )

        self.cost_functions_for_loss_functions[cost_function.loss_function].append(
            cost_function
        )

        self._add_batched_cost_function(cost_function, use_batches)

        if self.optim_vars.keys() & self.aux_vars.keys():
            raise ValueError(
                "Objective does not support a variable being both "
                "an optimization variable and an auxiliary variable."
            )

        self._is_setup = False

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
        is_loss_function: bool = False,
    ):
        if optim_vars:
            fn_var_list = function.optim_vars
            if is_cost_weight:
                self_vars_of_this_type = self.cost_weight_optim_vars
            elif is_loss_function:
                self_vars_of_this_type = self.loss_function_optim_vars
            else:
                self_vars_of_this_type = self.optim_vars
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
                del self._all_variables[variable.name]

                if self_vars_of_this_type is self.optim_vars:
                    batch_name = self._get_cost_function_variable_batch_name(variable)
                    variable_batch, variables = self.batched_optim_vars[batch_name]
                    variable_batch.data = None
                    del variables[variable.name]

                    if not variables:
                        del self.batched_optim_vars[batch_name]

    def _erase_batched_cost_function(self, cost_function: CostFunction):
        if cost_function.name in self.unbatched_cost_functions:
            del self.unbatched_cost_functions[cost_function.name]
        else:
            batch_name = self.batch_names_for_cost_functions[cost_function.name]
            cost_function_batch, cost_functions = self.batched_cost_functions[
                batch_name
            ]
            cost_fn_idx = cost_functions.index(cost_function)
            del cost_functions[cost_fn_idx]

            if len(cost_functions) == 0:
                del self.batched_cost_functions[batch_name]
            else:
                for var_name in cost_function_batch._optim_vars_attr_names:
                    setattr(cost_function_batch, var_name, None)

                for var_name in cost_function_batch._aux_vars_attr_names:
                    setattr(cost_function_batch, var_name, None)

            del self.batch_names_for_cost_functions[cost_function.name]

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

            # delete cost function from list of cost functions connected to its loss function
            loss_function = cost_function.loss_function
            cost_fn_idx = self.cost_functions_for_loss_functions[loss_function].index(
                cost_function
            )
            del self.cost_functions_for_loss_functions[loss_function][cost_fn_idx]

            # No more cost functions associated to this loss function, so can also delete
            if len(self.cost_functions_for_loss_functions[loss_function]) == 0:
                # erase its variables (if needed)
                self._erase_function_variables(
                    loss_function, optim_vars=True, is_loss_function=True
                )
                self._erase_function_variables(
                    loss_function, optim_vars=False, is_loss_function=True
                )
                del self.cost_functions_for_loss_functions[loss_function]

            self._erase_batched_cost_function(cost_function)

            # finally, delete the cost function
            del self.cost_functions[name]

            self._is_setup = False
        else:
            warnings.warn(
                "ddThis cost function is not in the objective, nothing to be done."
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

    # whether the objective is setup
    @property
    def is_setup(self) -> bool:
        return self._is_setup

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
        error_vector = torch.zeros(self.batch_size, self.dim()).to(
            device=self.device, dtype=self.dtype
        )
        pos = 0
        for cost_function_batch, cost_functions in self.batched_cost_functions.values():
            batch_errors = cost_function_batch.error()
            # TODO: Implement FuncTorch
            batch_pos = 0
            for cost_function in cost_functions:
                weighted_error = cost_function.weight_error(
                    batch_errors[batch_pos : batch_pos + self.batch_size]
                )
                error_vector[:, pos : pos + cost_function.dim()] = weighted_error
                batch_pos += self.batch_size
                pos += cost_function.dim()

        for cost_function in self.unbatched_cost_functions.values():
            error_vector[
                :, pos : pos + cost_function.dim()
            ] = cost_function.weighted_error()
            pos += cost_function.dim()
        if not also_update:
            self.update(old_data)
        return error_vector

    def error_squared_norm(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        also_update: bool = False,
    ) -> torch.Tensor:
        return (self.error(input_data=input_data, also_update=also_update) ** 2).sum(
            dim=1
        )

    def function_value(
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
        function_value_vector = torch.zeros(
            self.batch_size, len(self.cost_functions)
        ).to(device=self.device, dtype=self.dtype)
        pos = 0
        for cost_function_batch, cost_functions in self.batched_cost_functions.values():
            batch_errors = cost_function_batch.error()
            # TODO: Implement FuncTorch
            batch_pos = 0
            for cost_function in cost_functions:
                function_value_vector[
                    :, pos : pos + 1
                ] = cost_function.evaluate_function_value(
                    batch_errors[batch_pos : batch_pos + self.batch_size]
                )
                batch_pos += self.batch_size
                pos += 1

        for cost_function in self.unbatched_cost_functions.values():
            function_value_vector[:, pos : pos + 1] = cost_function.function_value()
            pos += 1
        if not also_update:
            self.update(old_data)
        return function_value_vector

    def objective_value(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        also_update: bool = False,
    ) -> torch.Tensor:
        return self.function_value(input_data=input_data, also_update=also_update).sum(
            dim=1
        )

    def copy(self) -> "Objective":
        new_objective = Objective(dtype=self.dtype)

        # First copy all individual cost weights
        old_to_new_cost_weight_map: Dict[CostWeight, CostWeight] = {}
        for cost_weight in self.cost_functions_for_weights:
            new_cost_weight = cost_weight.copy(
                new_name=cost_weight.name, keep_variable_names=True
            )
            old_to_new_cost_weight_map[cost_weight] = new_cost_weight

        # Then copy all individual loss functions
        old_to_new_loss_function_map: Dict[LossFunction, LossFunction] = {}
        for loss_function in self.cost_functions_for_loss_functions:
            new_loss_function = loss_function.copy(
                new_name=loss_function.name, keep_variable_names=True
            )
            old_to_new_loss_function_map[loss_function] = new_loss_function

        # Now copy the cost functions and assign the corresponding cost weight copy
        new_cost_functions: List[CostFunction] = []
        for cost_function in self.cost_functions.values():
            new_cost_function = cost_function.copy(
                new_name=cost_function.name, keep_variable_names=True
            )
            # we assign the allocated weight copies to avoid saving duplicates
            new_cost_function.weight = old_to_new_cost_weight_map[cost_function.weight]
            new_cost_function.loss_function = old_to_new_loss_function_map[
                cost_function.loss_function
            ]
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
            # LossFunction
            for i, var in enumerate(cost_function.loss_function.optim_vars):
                if var.name in new_objective.loss_function_optim_vars:
                    cost_function.loss_function.set_optim_var_at(
                        i, new_objective.loss_function_optim_vars[var.name]
                    )
            for i, aux_var in enumerate(cost_function.loss_function.aux_vars):
                if new_objective.has_aux_var(aux_var.name):
                    cost_function.loss_function.set_aux_var_at(
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

    def _update_variables(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        keep_batch: bool = False,
    ):
        input_data = input_data or {}
        for var_name, data in input_data.items():
            if data.ndim < 2:
                raise ValueError(
                    f"Input data tensors must have a batch dimension and "
                    f"one ore more data dimensions, but data.ndim={data.ndim} for "
                    f"tensor with name {var_name}."
                )
            if var_name in self.optim_vars:
                self.optim_vars[var_name].update(data, keep_batch=keep_batch)
            elif var_name in self.aux_vars:
                self.aux_vars[var_name].update(data, keep_batch=keep_batch)
            elif var_name in self.cost_weight_optim_vars:
                self.cost_weight_optim_vars[var_name].update(
                    data, keep_batch=keep_batch
                )
                warnings.warn(
                    "Updated a variable declared as optimization, but it is "
                    "only associated to cost weights and not to any cost functions. "
                    "Theseus optimizers will only update optimization variables "
                    "that are associated to one or more cost functions."
                )
            elif var_name in self.loss_function_optim_vars:
                self.loss_function_optim_vars[var_name].update(
                    data, keep_batch=keep_batch
                )
                warnings.warn(
                    "Updated a variable declared as optimization, but it is "
                    "only associated to loss functions and not to any cost functions. "
                    "Theseus optimizers will only update optimization variables "
                    "that are associated to one or more cost functions."
                )
            else:
                warnings.warn(
                    f"Attempted to update a tensor with name {var_name}, "
                    "which is not associated to any variable in the objective."
                )
        if len(input_data) != 0:
            self._is_setup = keep_batch

    def _update_batched_cost_functions(self):
        # Update batched cost functions for batch processing
        for batch_name, (
            cost_function_batch,
            cost_functions,
        ) in self.batched_cost_functions.items():
            vars_attr_name_lists = [
                cost_function_batch._optim_vars_attr_names,
                cost_function_batch._aux_vars_attr_names,
            ]
            for vars_attr_names in vars_attr_name_lists:
                for var_attr_name in vars_attr_names:
                    variable_batch = cast(
                        Variable, getattr(cost_function_batch, var_attr_name)
                    )
                    variable_batch_data = [
                        cast(Variable, getattr(cost_function, var_attr_name)).data
                        for cost_function in cost_functions
                    ]
                    variable_batch.data = torch.cat(variable_batch_data, dim=0)

                    if variable_batch.shape[0] != len(cost_functions) * self.batch_size:
                        raise ValueError(
                            f"Provided data for {var_attr_name} in batched cost function "
                            f"{batch_name} can not be batched."
                        )

    def _update_batched_optim_variables(self):
        # Update batched optim variables for batch processing
        for batch_name, (
            variable_batch,
            variables,
        ) in self.batched_optim_vars.items():
            variable_batch_data = [variable.data for variable in variables.values()]

            variable_batch.data = torch.cat(variable_batch_data, dim=0)

            if variable_batch.shape[0] != len(variables) * self.batch_size:
                raise ValueError(
                    f"Provided data for batched variable {batch_name} can not be batched."
                )

    def setup(self, input_data: Optional[Dict[str, torch.Tensor]] = None):
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

        self._update_variables(input_data=input_data, keep_batch=False)

        # Check that the batch size of all data is consistent after update
        batch_sizes = [v.data.shape[0] for v in self.optim_vars.values()]
        batch_sizes.extend([v.data.shape[0] for v in self.aux_vars.values()])
        self._batch_size = _get_batch_size(batch_sizes)

        with torch.enable_grad():
            self._update_batched_cost_functions()
            self._update_batched_optim_variables()

        self._is_setup = True

    def update(self, input_data: Optional[Dict[str, torch.Tensor]] = None):
        if self.is_setup:
            self._update_variables(input_data=input_data, keep_batch=True)
            if input_data is not None and len(input_data) != 0:
                with torch.enable_grad():
                    self._update_batched_cost_functions()
                    self._update_batched_optim_variables()
        else:
            self.setup(input_data)

    # iterates over cost functions
    def __iter__(self):
        return iter([f for f in self.cost_functions.values()])

    # Applies to() with given args to all tensors in the objective
    def to(self, *args, **kwargs):
        for cost_function in self.cost_functions.values():
            cost_function.to(*args, **kwargs)
        if self._is_setup:
            with torch.enable_grad():
                self._update_batched_cost_functions()
                self._update_batched_optim_variables()
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device or self.device
        self.dtype = dtype or self.dtype
