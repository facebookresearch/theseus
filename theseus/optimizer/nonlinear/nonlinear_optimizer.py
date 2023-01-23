# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from theseus.core import Objective
from theseus.optimizer import Optimizer, OptimizerInfo


class BackwardMode(Enum):
    UNROLL = 0
    IMPLICIT = 1
    TRUNCATED = 2
    DLM = 3

    @staticmethod
    def resolve(key: Union[str, "BackwardMode"]) -> "BackwardMode":
        if isinstance(key, BackwardMode):
            return key

        if not isinstance(key, str):
            raise ValueError("Backward mode must be th.BackwardMode or string.")

        try:
            backward_mode = BackwardMode[key.upper()]
        except KeyError:
            raise ValueError(
                f"Unrecognized backward mode f{key}."
                f"Valid choices are unroll, implicit, truncated, dlm."
            )
        return backward_mode


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
# to be optimized (e.g., sum of squared costs for NLLS), in addition to
# `_optimize_impl()` as defined by the base `Optimizer` class.
class NonlinearOptimizer(Optimizer, abc.ABC):
    _MAX_ALL_REJECT_ATTEMPTS = 3

    def __init__(
        self,
        objective: Objective,
        *args,
        vectorize: bool = False,
        abs_err_tolerance: float = 1e-8,
        rel_err_tolerance: float = 1e-5,
        max_iterations: int = 20,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(objective, vectorize=vectorize, **kwargs)
        self.params = NonlinearOptimizerParams(
            abs_err_tolerance, rel_err_tolerance, max_iterations, step_size
        )

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
        for var in self.objective.optim_vars.values():
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
            for var in self.objective.optim_vars.values():
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

    # Modifies the (no grad) info in place to add data of grad loop info
    def _merge_infos(
        self,
        grad_loop_info: NonlinearOptimizerInfo,
        num_no_grad_iters: int,
        num_grad_iters: int,
        info: NonlinearOptimizerInfo,
    ):
        total_iters = num_no_grad_iters + num_grad_iters
        # we add + 1 to all indices to account for the initial values
        info_idx = slice(num_no_grad_iters + 1, total_iters + 1)
        grad_info_idx = slice(1, num_grad_iters + 1)
        # Concatenate error histories
        if info.err_history is not None:
            info.err_history[:, info_idx] = grad_loop_info.err_history[:, grad_info_idx]
        if info.state_history is not None:
            for var in self.objective.optim_vars.values():
                info.state_history[var.name][
                    ..., info_idx
                ] = grad_loop_info.state_history[var.name][..., grad_info_idx]

        # Merge best solution and best error
        if info.best_solution is not None:
            best_solution = {}
            best_err_no_grad = info.best_err
            best_err_grad = grad_loop_info.best_err
            idx_no_grad = (best_err_no_grad < best_err_grad).cpu().view(-1, 1)
            best_err = torch.minimum(best_err_no_grad, best_err_grad)
            for var_name in info.best_solution:
                sol_no_grad = info.best_solution[var_name]
                sol_grad = grad_loop_info.best_solution[var_name]
                best_solution[var_name] = torch.where(
                    idx_no_grad, sol_no_grad, sol_grad
                )
            info.best_solution = best_solution
            info.best_err = best_err

        # Merge the converged status into the info from the detached loop,
        M = info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
        assert np.all(
            (grad_loop_info.status[M] == NonlinearOptimizerStatus.MAX_ITERATIONS)
            | (grad_loop_info.status[M] == NonlinearOptimizerStatus.CONVERGED)
        )
        info.status[M] = grad_loop_info.status[M]
        info.converged_iter[M] = (
            info.converged_iter[M] + grad_loop_info.converged_iter[M]
        )
        # If didn't coverge in either loop, remove misleading converged_iter value
        info.converged_iter[
            M & (grad_loop_info.status == NonlinearOptimizerStatus.MAX_ITERATIONS)
        ] = -1

    # Returns how many iterations to do with and without autograd
    def _split_backward_iters(self, **kwargs) -> Tuple[int, int]:
        if kwargs["backward_mode"] == BackwardMode.TRUNCATED:
            if "backward_num_iterations" not in kwargs:
                raise ValueError("backward_num_iterations expected but not received.")
            if kwargs["backward_num_iterations"] > self.params.max_iterations:
                warnings.warn(
                    f"Input backward_num_iterations "
                    f"(={kwargs['backward_num_iterations']}) > "
                    f"max_iterations (={self.params.max_iterations}). "
                    f"Using backward_num_iterations=max_iterations."
                )
            backward_num_iters = min(
                kwargs["backward_num_iterations"], self.params.max_iterations
            )
        else:
            backward_num_iters = {
                BackwardMode.UNROLL: self.params.max_iterations,
                BackwardMode.DLM: self.params.max_iterations,
                BackwardMode.IMPLICIT: 1,
            }[kwargs["backward_mode"]]
        return backward_num_iters, self.params.max_iterations - backward_num_iters
