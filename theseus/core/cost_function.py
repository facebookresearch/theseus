# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, List, Optional, Tuple, cast

import torch
import torch.autograd.functional as autogradF
from typing_extensions import Protocol

from .cost_weight import CostWeight, ScaleCostWeight
from .theseus_function import TheseusFunction
from .variable import Variable


# A cost function is defined by the variables interacting in it,
# and a cost weight for weighting errors and jacobians
# This is an abstract class for cost functions of different types.
# Each concrete function must implement an error method and the
# jacobian of the error, by implementing abstract methods
# `error` and `jacobians`, respectively.
class CostFunction(TheseusFunction, abc.ABC):
    def __init__(
        self,
        cost_weight: CostWeight,
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name)
        self.weight = cost_weight

    @abc.abstractmethod
    def error(self) -> torch.Tensor:
        pass

    # Returns (jacobians, error)
    @abc.abstractmethod
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass

    @abc.abstractmethod
    def dim(self) -> int:
        pass

    def weighted_error(self) -> torch.Tensor:
        error = self.error()
        return self.weight.weight_error(error)

    def weighted_jacobians_error(
        self,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        jacobian, err = self.jacobians()
        return self.weight.weight_jacobians_and_error(jacobian, err)

    # Must copy everything
    @abc.abstractmethod
    def _copy_impl(self, new_name: Optional[str] = None) -> "CostFunction":
        pass

    # Here for mypy compatibility downstream
    def copy(
        self, new_name: Optional[str] = None, keep_variable_names: bool = False
    ) -> "CostFunction":
        return cast(
            CostFunction,
            super().copy(new_name=new_name, keep_variable_names=keep_variable_names),
        )

    # calls to() on the cost weight, variables and any internal tensors
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight.to(*args, **kwargs)


# Function protocol for learnable cost functions. `optim_vars` and `aux_vars` are
# separated to facilitate internal autograd handling of optimization variables
class ErrFnType(Protocol):
    def __call__(
        self, optim_vars: Tuple[Variable, ...], aux_vars: Tuple[Variable, ...]
    ) -> torch.Tensor:
        ...


# The error function is assumed to receive variables in the format
#    err_fn(
#       optim_vars=(optim_vars[0].data, ..., optim_vars[N - 1].data),
#       aux_vars=(aux_vars[0].data, ..., aux_vars[M -1].data)
#   )
#
# The user also needs to explicitly specify the output's dimension
class AutoDiffCostFunction(CostFunction):
    def __init__(
        self,
        optim_vars: List[Variable],
        err_fn: ErrFnType,
        dim: int,
        cost_weight: Optional[CostWeight] = None,
        aux_vars: Optional[List[Variable]] = None,
        name: Optional[str] = None,
        autograd_strict: bool = False,
        autograd_vectorize: bool = False,
    ):
        if cost_weight is None:
            cost_weight = ScaleCostWeight(1.0)
        super().__init__(cost_weight, name=name)
        # this avoids doing aux_vars=[], which is a bad default since [] is mutable
        aux_vars = aux_vars or []

        def _register_vars_in_list(var_list_, is_optim=False):
            for var_ in var_list_:
                if hasattr(self, var_.name):
                    raise RuntimeError(f"Variable name {var_.name} is not allowed.")
                setattr(self, var_.name, var_)
                if is_optim:
                    self.register_optim_var(var_.name)
                else:
                    self.register_aux_var(var_.name)

        if len(optim_vars) < 1:
            raise ValueError(
                "AutodiffCostFunction must receive at least one optimization variable."
            )
        _register_vars_in_list(optim_vars, is_optim=True)
        _register_vars_in_list(aux_vars, is_optim=False)

        self._err_fn = err_fn
        self._dim = dim
        self._autograd_strict = autograd_strict
        self._autograd_vectorize = autograd_vectorize

        # The following are auxiliary Variable objects to hold tensor data
        # during jacobian computation without modifying the original Variable objects
        self._tmp_optim_vars = tuple(Variable(data=v.data) for v in optim_vars)

    def _compute_error(
        self,
    ) -> Tuple[torch.Tensor, Tuple[Variable, ...], Tuple[Variable, ...]]:
        optim_vars = tuple(v for v in self.optim_vars)
        aux_vars = tuple(v for v in self.aux_vars)
        err = self._err_fn(optim_vars=optim_vars, aux_vars=aux_vars)
        if err.shape[1] != self.dim():
            raise ValueError(
                "Output dimension of given error function doesn't match self.dim()."
            )
        return err, optim_vars, aux_vars

    def error(self) -> torch.Tensor:
        return self._compute_error()[0]

    # Returns (jacobians, error)
    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        err, optim_vars, aux_vars = self._compute_error()

        # this receives a list of torch tensors with data to set for tmp_optim_vars
        def jac_fn(*optim_vars_data_):
            assert len(optim_vars_data_) == len(self._tmp_optim_vars)
            for i, tensor in enumerate(optim_vars_data_):
                self._tmp_optim_vars[i].update(tensor)

            return self._err_fn(optim_vars=self._tmp_optim_vars, aux_vars=aux_vars)

        jacobians_full = autogradF.jacobian(
            jac_fn,
            tuple(v.data for v in optim_vars),
            create_graph=True,
            strict=self._autograd_strict,
            vectorize=self._autograd_vectorize,
        )
        aux_idx = torch.arange(err.shape[0])  # batch_size

        # torch autograd returns shape (batch_size, dim, batch_size, var_dim), which
        # includes derivatives of batches against each other.
        # this indexing recovers only the derivatives wrt the same batch
        jacobians = list(jac[aux_idx, :, aux_idx, :] for jac in jacobians_full)
        return jacobians, err

    def dim(self) -> int:
        return self._dim

    def _copy_impl(self, new_name: Optional[str] = None) -> "AutoDiffCostFunction":
        return AutoDiffCostFunction(
            [v.copy() for v in self.optim_vars],
            self._err_fn,
            self._dim,
            aux_vars=[v.copy() for v in self.aux_vars],
            cost_weight=self.weight.copy(),
            name=new_name,
        )

    def to(self, *args, **kwargs):
        # calls to() on the cost weight, variables and any internal tensors
        super().to(*args, **kwargs)
        for var in self._tmp_optim_vars:
            var.to(*args, **kwargs)
