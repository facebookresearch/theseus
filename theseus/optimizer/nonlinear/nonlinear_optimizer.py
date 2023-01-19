# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

import numpy as np
import torch

from theseus.core import Objective
from theseus.optimizer import Linearization, Optimizer, OptimizerInfo
from theseus.optimizer.linear import LinearSolver


@dataclass
class NonlinearOptimizerParams:
    abs_err_tolerance: float
    rel_err_tolerance: float
    max_iterations: int
    step_size: float

    def update(self, params_dict):
        for param, value in params_dict.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid nonlinear optimizer parameter {param}.")


class NonlinearOptimizerStatus(Enum):
    START = 0
    CONVERGED = 1
    MAX_ITERATIONS = 2
    FAIL = -1


# All info information is batched
@dataclass
class NonlinearOptimizerInfo(OptimizerInfo):
    converged_iter: torch.Tensor
    best_iter: torch.Tensor
    err_history: Optional[torch.Tensor]
    last_err: torch.Tensor
    best_err: torch.Tensor
    state_history: Optional[Dict[str, torch.Tensor]]  # variable name to state history


# Base class for all nonlinear optimizers.
# Provides methods useful for bookkeeping during optimization.
# Subclasses need to provide `error_metric()`, which computes the error to
# to optimized (e.g., sum of squared costs for NLLS), in addition to
# `_optimize_impl()` as defined by the base `Optimizer` class.
class NonlinearOptimizer(Optimizer, abc.ABC):
    _MAX_ALL_REJECT_ATTEMPTS = 3

    def __init__(
        self,
        objective: Objective,
        linear_solver_cls: Type[LinearSolver],
        *args,
        vectorize: bool = False,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        linear_solver_kwargs: Optional[Dict[str, Any]] = None,
        abs_err_tolerance: float = 1e-8,
        rel_err_tolerance: float = 1e-5,
        max_iterations: int = 20,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(objective, vectorize=vectorize, **kwargs)
        linear_solver_kwargs = linear_solver_kwargs or {}
        self.linear_solver = linear_solver_cls(
            objective,
            linearization_cls=linearization_cls,
            linearization_kwargs=linearization_kwargs,
            **linear_solver_kwargs,
        )
        self.ordering = self.linear_solver.linearization.ordering
        self.params = NonlinearOptimizerParams(
            abs_err_tolerance, rel_err_tolerance, max_iterations, step_size
        )
        self._tmp_optim_vars = tuple(v.copy(new_name=v.name) for v in self.ordering)

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    @abc.abstractmethod
    def _error_metric(
        self,
        input_tensors: Optional[Dict[str, torch.Tensor]] = None,
        also_update: bool = False,
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def _check_convergence(self, err: torch.Tensor, last_err: torch.Tensor):
        if err.abs().mean() < self.params.abs_err_tolerance:
            return torch.ones_like(err).bool()

        abs_error = (last_err - err).abs()
        rel_error = abs_error / last_err
        return (abs_error < self.params.abs_err_tolerance).logical_or(
            rel_error < self.params.rel_err_tolerance
        )

    def _maybe_init_best_solution(
        self, do_init: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not do_init:
            return None
        solution_dict = {}
        for var in self.ordering:
            solution_dict[var.name] = var.tensor.detach().clone().cpu()
        return solution_dict

    def _init_info(
        self,
        track_best_solution: bool,
        track_err_history: bool,
        track_state_history: bool,
    ) -> NonlinearOptimizerInfo:
        with torch.no_grad():
            last_err = self._error_metric()
        best_err = last_err.clone() if track_best_solution else None
        if track_err_history:
            err_history = (
                torch.ones(self.objective.batch_size, self.params.max_iterations + 1)
                * math.inf
            )
            assert last_err.grad_fn is None
            err_history[:, 0] = last_err.clone().cpu()
        else:
            err_history = None

        if track_state_history:
            state_history = {}
            for var in self.objective.optim_vars.values():
                state_history[var.name] = (
                    torch.ones(
                        self.objective.batch_size,
                        *var.shape[1:],
                        self.params.max_iterations + 1,
                    )
                    * math.inf
                )
                state_history[var.name][..., 0] = var.tensor.detach().clone().cpu()
        else:
            state_history = None

        return NonlinearOptimizerInfo(
            best_solution=self._maybe_init_best_solution(do_init=track_best_solution),
            last_err=last_err,
            best_err=best_err,
            status=np.array(
                [NonlinearOptimizerStatus.START] * self.objective.batch_size
            ),
            converged_iter=torch.zeros_like(last_err, dtype=torch.long),
            best_iter=torch.zeros_like(last_err, dtype=torch.long),
            err_history=err_history,
            state_history=state_history,
        )

    def _update_state_history(self, iter_idx: int, info: NonlinearOptimizerInfo):
        for var in self.objective.optim_vars.values():
            info.state_history[var.name][..., iter_idx + 1] = (
                var.tensor.detach().clone().cpu()
            )

    def _update_info(
        self,
        info: NonlinearOptimizerInfo,
        current_iter: int,
        err: torch.Tensor,
        converged_indices: torch.Tensor,
    ):
        info.converged_iter += 1 - converged_indices.long()
        if info.err_history is not None:
            assert not torch.is_grad_enabled()
            info.err_history[:, current_iter + 1] = err.clone().cpu()
        if info.state_history is not None:
            self._update_state_history(current_iter, info)

        if info.best_solution is not None:
            # Only copy best solution if needed (None means track_best_solution=False)
            assert info.best_err is not None
            good_indices = err < info.best_err
            info.best_iter[good_indices] = current_iter
            for var in self.ordering:
                info.best_solution[var.name][good_indices] = (
                    var.tensor.detach().clone()[good_indices].cpu()
                )

            info.best_err = torch.minimum(info.best_err, err)

        converged_indices = self._check_convergence(err, info.last_err)
        info.status[
            np.array(converged_indices.detach().cpu())
        ] = NonlinearOptimizerStatus.CONVERGED

    @abc.abstractmethod
    def _optimize_impl(self, **kwargs) -> OptimizerInfo:
        pass
