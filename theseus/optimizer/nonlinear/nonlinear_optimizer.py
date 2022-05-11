# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type, cast

import numpy as np
import torch

import theseus.constants
from theseus.core import Objective, Variable
from theseus.optimizer import Linearization, Optimizer, OptimizerInfo
from theseus.optimizer.linear import LinearSolver, LUCudaSparseSolver


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


class BackwardMode(Enum):
    FULL = 0
    IMPLICIT = 1
    TRUNCATED = 2
    DLM = 3


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
        self, track_best_solution: bool, track_err_history: bool, verbose: bool
    ) -> NonlinearOptimizerInfo:
        with torch.no_grad():
            last_err = self.objective.objective_value() / 2
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
            assert err.grad_fn is None
            info.err_history[:, current_iter + 1] = err.clone().cpu()

        if info.best_solution is not None:
            # Only copy best solution if needed (None means track_best_solution=False)
            assert info.best_err is not None
            good_indices = err < info.best_err
            info.best_iter[good_indices] = current_iter
            for var in self.linear_solver.linearization.ordering:
                info.best_solution[var.name][good_indices] = (
                    var.data.detach().clone()[good_indices].cpu()
                )

            info.best_err = torch.minimum(info.best_err, err)

        converged_indices = self._check_convergence(err, info.last_err)
        info.status[
            np.array(converged_indices.detach().cpu())
        ] = NonlinearOptimizerStatus.CONVERGED

    # Modifies the (no grad) info in place to add data of grad loop info
    def _merge_infos(
        self,
        grad_loop_info: NonlinearOptimizerInfo,
        num_no_grad_iter: int,
        backward_num_iterations: int,
        info: NonlinearOptimizerInfo,
    ):
        # Concatenate error histories
        if info.err_history is not None:
            info.err_history[:, num_no_grad_iter:] = grad_loop_info.err_history[
                :, : backward_num_iterations + 1
            ]
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

    # loop for the iterative optimizer
    def _optimize_loop(
        self,
        start_iter: int,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        truncated_grad_loop: bool,
        grouped_retract: bool = True,
        **kwargs,
    ) -> NonlinearOptimizerInfo:
        converged_indices = torch.zeros_like(info.last_err).bool()
        for it_ in range(start_iter, start_iter + num_iter):
            # do optimizer step
            self.linear_solver.linearization.linearize()
            try:
                if truncated_grad_loop:
                    # The derivation for implicit differentiation states that
                    # the autograd-enabled loop (which `truncated_grad_loop` signals)
                    # must be done using Gauss-Newton steps. Well, technically,
                    # full Newton, but it seems less stable numerically and
                    # GN is working well so far.
                    delta = self.linear_solver.solve()
                else:
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

            if truncated_grad_loop:
                # This is a "secret" option that is currently being tested in the
                # context of implicit differentiation. Might be added as a supported
                # kwarg in the future with a different name, or removed altogether.
                if "__keep_final_step_size__" in kwargs:
                    step_size = self.params.step_size
                else:
                    step_size = 1.0
                force_update = True
            else:
                step_size = self.params.step_size
                force_update = False

            self.retract_and_update_variables(
                delta,
                converged_indices,
                step_size,
                force_update=force_update,
                grouped_retract=grouped_retract,
            )

            # check for convergence
            with torch.no_grad():
                err = self.objective.objective_value() / 2
                self._update_info(info, it_, err, converged_indices)
                if verbose:
                    print(
                        f"Nonlinear optimizer. Iteration: {it_+1}. "
                        f"Error: {err.mean().item()}"
                    )
                converged_indices = self._check_convergence(err, info.last_err)
                info.status[
                    converged_indices.cpu().numpy()
                ] = NonlinearOptimizerStatus.CONVERGED
                if converged_indices.all():
                    break  # nothing else will happen at this point
                info.last_err = err

        info.status[
            info.status == NonlinearOptimizerStatus.START
        ] = NonlinearOptimizerStatus.MAX_ITERATIONS
        return info

    # `track_best_solution` keeps a **detached** copy (as in no gradient info)
    # of the best variables found, but it is optional to avoid unnecessary copying
    # if this is not needed
    def _optimize_impl(
        self,
        track_best_solution: bool = False,
        track_err_history: bool = False,
        verbose: bool = False,
        backward_mode: BackwardMode = BackwardMode.FULL,
        grouped_retract: bool = True,
        **kwargs,
    ) -> OptimizerInfo:
        with torch.no_grad():
            info = self._init_info(track_best_solution, track_err_history, verbose)

        if (
            type(self.linear_solver) is LUCudaSparseSolver
            and self.linear_solver.batch_size != self.objective.batch_size
        ):
            self.linear_solver.reset(self.objective.batch_size)

        if verbose:
            print(
                f"Nonlinear optimizer. Iteration: 0. "
                f"Error: {info.last_err.mean().item()}"
            )

        if backward_mode in [BackwardMode.FULL, BackwardMode.DLM]:
            info = self._optimize_loop(
                start_iter=0,
                num_iter=self.params.max_iterations,
                info=info,
                verbose=verbose,
                truncated_grad_loop=False,
                grouped_retract=grouped_retract,
                **kwargs,
            )
            # If didn't coverge, remove misleading converged_iter value
            info.converged_iter[
                info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
            ] = -1
            return info
        elif backward_mode in [BackwardMode.IMPLICIT, BackwardMode.TRUNCATED]:
            if backward_mode == BackwardMode.IMPLICIT:
                backward_num_iterations = 1
            else:
                if "backward_num_iterations" not in kwargs:
                    raise ValueError(
                        "backward_num_iterations expected but not received"
                    )
                if kwargs["backward_num_iterations"] > self.params.max_iterations:
                    warnings.warn(
                        f"Input backward_num_iterations "
                        f"(={kwargs['backward_num_iterations']}) > "
                        f"max_iterations (={self.params.max_iterations}). "
                        f"Using backward_num_iterations=max_iterations."
                    )
                backward_num_iterations = min(
                    kwargs["backward_num_iterations"], self.params.max_iterations
                )

            num_no_grad_iter = self.params.max_iterations - backward_num_iterations
            with torch.no_grad():
                self._optimize_loop(
                    start_iter=0,
                    num_iter=num_no_grad_iter,
                    info=info,
                    verbose=verbose,
                    truncated_grad_loop=False,
                    grouped_retract=grouped_retract,
                    **kwargs,
                )

            grad_loop_info = self._init_info(
                track_best_solution, track_err_history, verbose
            )
            self._optimize_loop(
                start_iter=0,
                num_iter=backward_num_iterations,
                info=grad_loop_info,
                verbose=verbose,
                truncated_grad_loop=True,
                grouped_retract=grouped_retract,
                **kwargs,
            )

            # Adds grad_loop_info results to original info
            self._merge_infos(
                grad_loop_info, num_no_grad_iter, backward_num_iterations, info
            )

            return info
        else:
            raise ValueError("Unrecognized backward mode")

    @abc.abstractmethod
    def compute_delta(self, **kwargs) -> torch.Tensor:
        pass

    # retracts all variables in the given order and updates their values
    # with the result
    def retract_and_update_variables(
        self,
        delta: torch.Tensor,
        converged_indices: torch.Tensor,
        step_size: float,
        force_update: bool = False,
        grouped_retract: bool = True,
    ):
        var_idx = 0
        delta = step_size * delta

        if not grouped_retract:
            assert len(self.objective.batched_cost_functions) == 0
            for var in self.linear_solver.linearization.ordering:
                new_var = var.retract(delta[:, var_idx : var_idx + var.dof()])
                if force_update:
                    var.update(new_var.data)
                else:
                    var.update(new_var.data, batch_ignore_mask=converged_indices)
                var_idx += var.dof()
            return

        objective = self.objective
        # TODO: what does converged_indices do?
        for variable_batch, variables in self.objective.batched_optim_vars.values():
            cnts = variable_batch.dof() * len(variables)

            variable_batch.data = torch.cat(
                [variable.data for variable in variables.values()], dim=0
            )

            delta_batch = torch.cat(
                [
                    delta[:, idx : idx + variable_batch.dof()]
                    for idx in range(var_idx, var_idx + cnts, variable_batch.dof())
                ],
                dim=0,
            )
            new_variable_batch = variable_batch.retract(delta_batch)

            batch_pos = 0

            for variable in variables.values():
                new_variable = new_variable_batch[
                    batch_pos : batch_pos + objective.batch_size
                ]

                if force_update:
                    variable.update(new_variable)
                else:
                    variable.update(new_variable, batch_ignore_mask=converged_indices)

                batch_pos += objective.batch_size

            var_idx += cnts

        # update batched cost functions
        # TODO: Implement FuncTorch
        for (
            batch_cost_function,
            cost_functions,
            _,
        ) in objective.batched_cost_functions.values():
            vars_attr_names = batch_cost_function._optim_vars_attr_names
            for var_attr_name in vars_attr_names:
                batch_variable = cast(
                    Variable, getattr(batch_cost_function, var_attr_name)
                )
                batch_variable.data = torch.cat(
                    [
                        cast(Variable, getattr(cost_function, var_attr_name)).data
                        for cost_function in cost_functions
                    ],
                    dim=0,
                )
