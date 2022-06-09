# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from itertools import count
from typing import Generator, Iterable, List, Optional, Sequence

from theseus.geometry import Manifold

from .variable import Variable


# Base class that provides objective-tracking functionality for functions of tensors
# such as cost fuctions and cost weights. Variables are classified as either
# optimization variables (of the objective), or auxiliary variables.
#
# Subclasses must register their optimization variables and auxiliary variables
# with methods `register_optim_vars` and `register_aux_vars`, respectively.
class TheseusFunction(abc.ABC):
    _ids = count(0)

    def __init__(self, name: Optional[str] = None):
        self._id = next(self._ids)
        if name:
            self.name = name
        else:
            self.name = self.get_default_name()
        self._optim_vars_attr_names: List[str] = []
        self._aux_vars_attr_names: List[str] = []

    def get_default_name(self) -> str:
        return f"{self.__class__.__name__}__{self._id}"

    def register_optim_var(self, attr_name: str):
        if attr_name in self._optim_vars_attr_names:
            raise ValueError(
                "Tried to register an optimization variable that was previously "
                "registered."
            )
        if attr_name in self._aux_vars_attr_names:
            raise ValueError(
                "An optimization variable cannot also be an auxiliary variable."
            )
        self._optim_vars_attr_names.append(attr_name)

    def register_aux_var(self, attr_name: str):
        if attr_name in self._aux_vars_attr_names:
            raise ValueError(
                "Tried to register an auxiliary variable that was previously registered."
            )
        if attr_name in self._optim_vars_attr_names:
            raise ValueError(
                "An auxiliary variable cannot also be an optimization variable."
            )
        self._aux_vars_attr_names.append(attr_name)

    def register_optim_vars(self, variable_names: Sequence[str]):
        for name in variable_names:
            self.register_optim_var(name)

    def register_aux_vars(self, aux_var_names: Sequence[str]):
        for name in aux_var_names:
            self.register_aux_var(name)

    def register_vars(self, vars: Iterable[Variable], is_optim_vars: bool = False):
        for var in vars:
            if hasattr(self, var.name):
                raise RuntimeError(f"Variable name {var.name} is not allowed.")
            setattr(self, var.name, var)
            if is_optim_vars:
                self.register_optim_var(var.name)
            else:
                self.register_aux_var(var.name)

    # Must copy everything
    @abc.abstractmethod
    def _copy_impl(self, new_name: Optional[str] = None) -> "TheseusFunction":
        pass

    def _has_duplicate_vars(self, another: "TheseusFunction") -> bool:
        def _check(base: Iterable[Variable], vectorized: Iterable[Variable]) -> bool:
            return len(set(base) & set(vectorized)) > 0

        return _check(self.optim_vars, another.optim_vars) | _check(
            self.aux_vars, another.aux_vars
        )

    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "TheseusFunction":
        if not new_name:
            new_name = f"{self.name}_copy"
        new_fn = self._copy_impl(new_name=new_name)
        if keep_variable_names:
            for old_var, new_var in zip(self.optim_vars, new_fn.optim_vars):
                new_var.name = old_var.name
            for old_aux, new_aux in zip(self.aux_vars, new_fn.aux_vars):
                new_aux.name = old_aux.name

        if self._has_duplicate_vars(new_fn):
            raise RuntimeError(
                f"{self.__class__.__name__}.copy() resulted in one of the original "
                "variables being reused. copy() requires all variables to be copied "
                "to new variables, so please re-implement _copy_impl() to satisfy this "
                "property."
            )

        return new_fn

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    @property
    def optim_vars(self) -> Generator[Manifold, None, None]:
        return (getattr(self, attr) for attr in self._optim_vars_attr_names)

    @property
    def aux_vars(self) -> Generator[Variable, None, None]:
        return (getattr(self, attr) for attr in self._aux_vars_attr_names)

    def optim_var_at(self, index: int) -> Manifold:
        return getattr(self, self._optim_vars_attr_names[index])

    def aux_var_at(self, index: int) -> Variable:
        return getattr(self, self._aux_vars_attr_names[index])

    # This method is only used when copying the full objective.
    # It replaces the variable at the given index in the interal list of
    # variables with the variable passed as argument.
    # This is used by methods `set_optim_var_at()` and `set_aux_var_at()`.
    def _set_variable_at(
        self,
        index: int,
        variable: Variable,
        variable_attrs_array: List,
        variable_type: str,
    ):
        if index >= len(variable_attrs_array):
            raise ValueError(f"{variable_type} index out of range.")
        setattr(self, variable_attrs_array[index], variable)

    # see `_set_variable_at()`
    def set_optim_var_at(self, index: int, variable: Manifold):
        self._set_variable_at(
            index, variable, self._optim_vars_attr_names, "Optimization variable"
        )

    # see `_set_variable_at()`
    def set_aux_var_at(self, index: int, variable: Variable):
        self._set_variable_at(
            index, variable, self._aux_vars_attr_names, "Auxiliary variable"
        )

    def num_optim_vars(self):
        return len(self._optim_vars_attr_names)

    def num_aux_vars(self):
        return len(self._aux_vars_attr_names)

    # calls to() on the cost weight, optimization variables and any internal tensors
    def to(self, *args, **kwargs):
        for var in self.optim_vars:
            var.to(*args, **kwargs)
        for aux in self.aux_vars:
            aux.to(*args, **kwargs)
