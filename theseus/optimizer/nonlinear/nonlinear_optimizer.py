# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

import numpy as np
import torch

import theseus.constants
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
                raise ValueError(f"Invalid nonlinear least squares parameter {param}.")


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


class NonlinearOptimizer(Optimizer, abc.ABC):
    def __init__(
        self,
        objective: Objective,
        linear_solver_cls: Type[LinearSolver],
        *args,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        linear_solver_kwargs: Optional[Dict[str, Any]] = None,
        abs_err_tolerance: float = 1e-10,
        rel_err_tolerance: float = 1e-8,
        max_iterations: int = 20,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(objective)
        linear_solver_kwargs = linear_solver_kwargs or {}
        self.linear_solver = linear_solver_cls(
            objective,
            linearization_cls=linearization_cls,
            linearization_kwargs=linearization_kwargs,
            **linear_solver_kwargs,
        )
        self.params = NonlinearOptimizerParams(
            abs_err_tolerance, rel_err_tolerance, max_iterations, step_size
        )

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def _check_convergence(self, err: torch.Tensor, last_err: torch.Tensor):
        assert not torch.is_grad_enabled()
        if err.abs().mean() < theseus.constants.EPS:
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
        for var in self.linear_solver.linearization.ordering:
            solution_dict[var.name] = var.data.detach().clone().cpu()
        return solution_dict

    def _init_info(
        self, last_err: torch.Tensor, track_best_solution: bool, verbose: bool
    ) -> NonlinearOptimizerInfo:
        if verbose:
            err_history = (
                torch.ones(self.objective.batch_size, self.params.max_iterations + 1)
                * math.inf
            )
            assert last_err.grad_fn is None
            err_history[:, 0] = last_err.clone().cpu()
        else:
            err_history = None
        return NonlinearOptimizerInfo(
            best_solution=self._maybe_init_best_solution(do_init=track_best_solution),
            status=np.array(
                [NonlinearOptimizerStatus.START] * self.objective.batch_size
            ),
            converged_iter=torch.zeros_like(last_err, dtype=torch.long),
            best_iter=torch.zeros_like(last_err, dtype=torch.long),
            err_history=err_history,
        )

        # Only copy best solution if needed (None means track_best_solution=False)

    def _update_info(
        self,
        info: NonlinearOptimizerInfo,
        current_iter: int,
        best_err: Optional[torch.Tensor],
        err: torch.Tensor,
        converged_indices: torch.Tensor,
    ) -> torch.Tensor:
        info.converged_iter += 1 - converged_indices.long()
        if info.err_history is not None:
            assert err.grad_fn is None
            info.err_history[:, current_iter + 1] = err.clone().cpu()
        if info.best_solution is None:
            return best_err
        # Only copy best solution if needed (None means track_best_solution=False)
        assert best_err is not None
        good_indices = err < best_err
        info.best_iter[good_indices] = current_iter
        for var in self.linear_solver.linearization.ordering:
            info.best_solution[var.name][good_indices] = (
                var.data.detach().clone()[good_indices].cpu()
            )
        return torch.minimum(best_err, err)

    # `track_best_solution` keeps a **detached** copy (as in no gradient info)
    # of the best variables found, but it is optional to avoid unnecessary copying
    # if this is not needed
    #
    # if verbose, info will also keep track of the full error history
    def _optimize_impl(
        self,
        track_best_solution: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> OptimizerInfo:
        # All errors are only used for stopping conditions, so they are outside
        # compute graph
        last_err = self.objective.error_squared_norm().detach() / 2

        if verbose:
            print(
                f"Nonlinear optimizer. Iteration: {0}. Error: {last_err.mean().item()}"
            )

        best_err = last_err.clone() if track_best_solution else None
        converged_indices = torch.zeros_like(last_err).bool()
        info = self._init_info(last_err, track_best_solution, verbose)
        for it_ in range(self.params.max_iterations):
            # do optimizer step
            self.linear_solver.linearization.linearize()
            try:
                delta = self.compute_delta(**kwargs)
            except RuntimeError as run_err:
                msg = (
                    f"There was an error while running the linear optimizer. "
                    f"Original error message: {run_err}."
                )
                if torch.is_grad_enabled():
                    raise RuntimeError(
                        msg + " Backward pass will not work. To obtain "
                        "the best solution seen before the error, run with "
                        "torch.no_grad()"
                    )
                else:
                    warnings.warn(msg, RuntimeWarning)
                    info.status[:] = NonlinearOptimizerStatus.FAIL
                    return info
            self.retract_and_update_variables(delta, converged_indices)

            # check for convergence
            with torch.no_grad():
                err = self.objective.error_squared_norm().detach() / 2
                best_err = self._update_info(
                    info, it_, best_err, err, converged_indices
                )
                if verbose:
                    print(
                        f"Nonlinear optimizer. Iteration: {it_ + 1}. "
                        f"Error: {err.mean().item()}"
                    )
                converged_indices = self._check_convergence(err, last_err)
                info.status[
                    converged_indices.cpu().numpy()
                ] = NonlinearOptimizerStatus.CONVERGED
                if converged_indices.all():
                    break  # nothing else will happen at this point
                last_err = err
        info.status[
            info.status == NonlinearOptimizerStatus.START
        ] = NonlinearOptimizerStatus.MAX_ITERATIONS
        return info

    @abc.abstractmethod
    def compute_delta(self, **kwargs) -> torch.Tensor:
        pass

    def compute_samples(self, n_samples: int, temperature: float) -> torch.Tensor:
        pass

    # retracts all variables in the given order and updates their values
    # with the result
    def retract_and_update_variables(
        self, delta: torch.Tensor, converged_indices: torch.Tensor
    ):
        var_idx = 0
        delta = self.params.step_size * delta
        for var in self.linear_solver.linearization.ordering:
            new_var = var.retract(delta[:, var_idx : var_idx + var.dof()])
            var.update(new_var.data, batch_ignore_mask=converged_indices)
            var_idx += var.dof()
