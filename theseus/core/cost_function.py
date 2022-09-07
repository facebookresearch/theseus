# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Callable, List, Optional, Sequence, Tuple, cast
import warnings

import torch
import torch.autograd.functional as autogradF
from typing_extensions import Protocol

from functorch import vmap, jacrev

from theseus.geometry import Manifold
from theseus.geometry.lie_group_check import no_lie_group_check

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
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._weight = cost_weight

    @property
    def weight(self) -> CostWeight:
        return self._weight

    @weight.setter
    def weight(self, weight: CostWeight):
        self._weight = weight

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
#       optim_vars=(optim_vars[0].tensor, ..., optim_vars[N - 1].tensor),
#       aux_vars=(aux_vars[0].tensor, ..., aux_vars[M -1].tensor)
#   )
#
# The user also needs to explicitly specify the output's dimension
class AutoDiffCostFunction(CostFunction):
    def __init__(
        self,
        optim_vars: Sequence[Manifold],
        err_fn: ErrFnType,
        dim: int,
        cost_weight: Optional[CostWeight] = None,
        aux_vars: Optional[Sequence[Variable]] = None,
        name: Optional[str] = None,
        autograd_strict: bool = False,
        autograd_vectorize: bool = False,
        autograd_loop_over_batch: bool = False,
        autograd_functorch: bool = False,
    ):
        if cost_weight is None:
            cost_weight = ScaleCostWeight(1.0)
        super().__init__(cost_weight, name=name)
        # this avoids doing aux_vars=[], which is a bad default since [] is mutable
        aux_vars = aux_vars or []

        if len(optim_vars) < 1:
            raise ValueError(
                "AutodiffCostFunction must receive at least one optimization variable."
            )
        self.register_vars(optim_vars, is_optim_vars=True)
        self.register_vars(aux_vars, is_optim_vars=False)

        self._err_fn = err_fn
        self._dim = dim
        self._autograd_strict = autograd_strict
        self._autograd_vectorize = autograd_vectorize

        # The following are auxiliary Variable objects to hold tensor data
        # during jacobian computation without modifying the original Variable objects
        self._tmp_optim_vars = tuple(v.copy() for v in optim_vars)
        self._tmp_aux_vars = None
        self._tmp_optim_vars_for_loop = None
        self._tmp_aux_vars_for_loop = None

        self._autograd_loop_over_batch = autograd_loop_over_batch
        self._autograd_functorch = autograd_functorch

        if autograd_functorch and self._autograd_loop_over_batch:
            self._autograd_loop_over_batch = False
            warnings.warn(
                "autograd_use_functorch=True overrides given autograd_loop_over_batch=True, "
                "so the latter will be set to False"
            )

        if self._autograd_loop_over_batch:
            self._tmp_optim_vars_for_loop = tuple(v.copy() for v in optim_vars)
            self._tmp_aux_vars_for_loop = tuple(v.copy() for v in aux_vars)

            for i, optim_var in enumerate(optim_vars):
                self._tmp_optim_vars_for_loop[i].update(optim_var.tensor)

            for i, aux_var in enumerate(aux_vars):
                self._tmp_aux_vars_for_loop[i].update(aux_var.tensor)

        if self._autograd_functorch:
            self._tmp_aux_vars = tuple(v.copy() for v in aux_vars)

    def _compute_error(
        self,
    ) -> Tuple[torch.Tensor, Tuple[Manifold, ...], Tuple[Variable, ...]]:
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

    def _make_jac_fn(
        self, tmp_optim_vars: Tuple[Manifold, ...], tmp_aux_vars: Tuple[Variable, ...]
    ) -> Callable:
        def jac_fn(*optim_vars_tensors_):
            assert len(optim_vars_tensors_) == len(tmp_optim_vars)
            for i, tensor in enumerate(optim_vars_tensors_):
                tmp_optim_vars[i].update(tensor)

            return self._err_fn(optim_vars=tmp_optim_vars, aux_vars=tmp_aux_vars)

        return jac_fn

    def _compute_autograd_jacobian(
        self, optim_tensors: Tuple[torch.Tensor, ...], jac_fn: Callable
    ) -> Tuple[torch.Tensor, ...]:
        return autogradF.jacobian(
            jac_fn,
            optim_tensors,
            create_graph=True,
            strict=self._autograd_strict,
            vectorize=self._autograd_vectorize,
        )

    def _make_jac_fn_functorch(
        self, tmp_optim_vars: Tuple[Manifold, ...], tmp_aux_vars: Tuple[Variable, ...]
    ):
        def jac_fn(optim_vars_tensors_, aux_vars_tensors_):
            assert len(optim_vars_tensors_) == len(tmp_optim_vars)

            # disable tensor checks and other operations that are incompatible with functorch
            with no_lie_group_check():
                for i, tensor in enumerate(optim_vars_tensors_):
                    tmp_optim_vars[i].update(tensor.unsqueeze(0))

                # only aux_var in current batch is evaluated
                for i, tensor in enumerate(aux_vars_tensors_):
                    tmp_aux_vars[i].update(tensor.unsqueeze(0))

                # return [0] since functorch expects no batch output
                return self._err_fn(optim_vars=tmp_optim_vars, aux_vars=tmp_aux_vars)[0]

        return jac_fn

    def _compute_autograd_jacobian_functorch(
        self,
        optim_tensors: Tuple[torch.Tensor, ...],
        aux_tensors: Tuple[torch.Tensor, ...],
        jac_fn: Callable,
    ) -> Tuple[torch.Tensor, ...]:
        return vmap(jacrev(jac_fn, argnums=0))(optim_tensors, aux_tensors)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        err, optim_vars, aux_vars = self._compute_error()
        if self._autograd_functorch:
            jacobians_full = self._compute_autograd_jacobian_functorch(
                tuple(v.tensor for v in optim_vars),
                tuple(v.tensor for v in aux_vars),
                self._make_jac_fn_functorch(self._tmp_optim_vars, self._tmp_aux_vars),
            )
        elif self._autograd_loop_over_batch:
            jacobians_raw_loop: List[Tuple[torch.Tensor, ...]] = []
            for n in range(optim_vars[0].shape[0]):
                for i, aux_var in enumerate(aux_vars):
                    self._tmp_aux_vars_for_loop[i].update(aux_var.tensor[n : n + 1])

                jacobians_n = self._compute_autograd_jacobian(
                    tuple(v.tensor[n : n + 1] for v in optim_vars),
                    self._make_jac_fn(
                        self._tmp_optim_vars_for_loop, self._tmp_aux_vars_for_loop
                    ),
                )
                jacobians_raw_loop.append(jacobians_n)

            jacobians_full = tuple(
                torch.cat([jac[k][:, :, 0, :] for jac in jacobians_raw_loop], dim=0)
                for k in range(len(optim_vars))
            )
        else:
            jacobians_raw = self._compute_autograd_jacobian(
                tuple(v.tensor for v in optim_vars),
                self._make_jac_fn(self._tmp_optim_vars, aux_vars),
            )
            aux_idx = torch.arange(err.shape[0])  # batch_size
            jacobians_full = tuple(jac[aux_idx, :, aux_idx, :] for jac in jacobians_raw)

        # torch autograd returns shape (batch_size, dim, batch_size, var_dim), which
        # includes derivatives of batches against each other.
        # this indexing recovers only the derivatives wrt the same batch
        jacobians = list(
            v.project(jac, is_sparse=True) for v, jac in zip(optim_vars, jacobians_full)
        )

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
            autograd_loop_over_batch=self._autograd_loop_over_batch,
            autograd_functorch=self._autograd_functorch,
        )

    def to(self, *args, **kwargs):
        # calls to() on the cost weight, variables and any internal tensors
        super().to(*args, **kwargs)
        for var in self._tmp_optim_vars:
            var.to(*args, **kwargs)

        if self._autograd_loop_over_batch:
            for var in self._tmp_optim_vars_for_loop:
                var.to(*args, **kwargs)

            for var in self._tmp_aux_vars_for_loop:
                var.to(*args, **kwargs)

        if self._autograd_functorch:
            for var in self._tmp_aux_vars:
                var.to(*args, **kwargs)
