# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, NoReturn, Optional, Type, Union

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


EndIterCallbackType = Callable[
    ["NonlinearOptimizer", NonlinearOptimizerInfo, torch.Tensor, int], NoReturn
]


class NonlinearOptimizer(Optimizer, abc.ABC):
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
        self.params = NonlinearOptimizerParams(
            abs_err_tolerance, rel_err_tolerance, max_iterations, step_size
        )

    def set_params(self, **kwargs):
        self.params.update(kwargs)

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
        for var in self.linear_solver.linearization.ordering:
            solution_dict[var.name] = var.tensor.detach().clone().cpu()
        return solution_dict

    def _init_info(
        self,
        track_best_solution: bool,
        track_err_history: bool,
        track_state_history: bool,
    ) -> NonlinearOptimizerInfo:
        with torch.no_grad():
            last_err = self.objective.error_squared_norm() / 2
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
            assert err.grad_fn is None
            info.err_history[:, current_iter + 1] = err.clone().cpu()
        if info.state_history is not None:
            self._update_state_history(current_iter, info)

        if info.best_solution is not None:
            # Only copy best solution if needed (None means track_best_solution=False)
            assert info.best_err is not None
            good_indices = err < info.best_err
            info.best_iter[good_indices] = current_iter
            for var in self.linear_solver.linearization.ordering:
                info.best_solution[var.name][good_indices] = (
                    var.tensor.detach().clone()[good_indices].cpu()
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

    # loop for the iterative optimizer
    def _optimize_loop(
        self,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        truncated_grad_loop: bool,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        **kwargs,
    ) -> int:
        steps_tensor: torch.Tensor = None  # type: ignore
        converged_indices = torch.zeros_like(info.last_err).bool()
        iters_done = 0
        for it_ in range(num_iter):
            iters_done += 1
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
                    return iters_done

            if truncated_grad_loop:
                # This is a "secret" option that is currently being tested in the
                # context of implicit differentiation. Might be added as a supported
                # kwarg in the future with a different name, or removed altogether.
                # If the option is present and set to True, we don't use step size = 1
                # 1 for the truncated steps, resulting in scaled gradients, but
                # possibly better solution.
                if not kwargs.get("__keep_final_step_size__", False):
                    steps_tensor = torch.ones_like(delta)
                force_update = True
            else:
                force_update = False

            with torch.no_grad():
                if steps_tensor is None:
                    steps_tensor = torch.ones_like(delta) * self.params.step_size
            self.objective.retract_optim_vars(
                delta * steps_tensor,
                self.linear_solver.linearization.ordering,
                ignore_mask=converged_indices,
                force_update=force_update,
            )

            # check for convergence
            with torch.no_grad():
                err = self.objective.error_squared_norm() / 2
                self._update_info(info, it_, err, converged_indices)
                self.update_optimizer_state(
                    last_err=info.last_err, new_err=err, delta=delta, **kwargs
                )
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

                if end_iter_callback is not None:
                    end_iter_callback(self, info, delta, it_)

        info.status[
            info.status == NonlinearOptimizerStatus.START
        ] = NonlinearOptimizerStatus.MAX_ITERATIONS
        return iters_done

    # `track_best_solution` keeps a **detached** copy (as in no gradient info)
    # of the best variables found, but it is optional to avoid unnecessary copying
    # if this is not needed
    #
    # If `end_iter_callback` is passed, it's called at the very end of each optimizer
    # iteration, with four input arguments: (optimizer, info, delta, step_idx).
    def _optimize_impl(
        self,
        track_best_solution: bool = False,
        track_err_history: bool = False,
        track_state_history: bool = False,
        verbose: bool = False,
        backward_mode: Union[str, BackwardMode] = BackwardMode.UNROLL,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        **kwargs,
    ) -> OptimizerInfo:
        self.reset(**kwargs)
        backward_mode = BackwardMode.resolve(backward_mode)
        with torch.no_grad():
            info = self._init_info(
                track_best_solution, track_err_history, track_state_history
            )

        if verbose:
            print(
                f"Nonlinear optimizer. Iteration: 0. "
                f"Error: {info.last_err.mean().item()}"
            )

        if backward_mode in [BackwardMode.UNROLL, BackwardMode.DLM]:
            self._optimize_loop(
                start_iter=0,
                num_iter=self.params.max_iterations,
                info=info,
                verbose=verbose,
                truncated_grad_loop=False,
                end_iter_callback=end_iter_callback,
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
                # actual_num_iters could be < num_iter due to early convergence
                no_grad_iters_done = self._optimize_loop(
                    num_iter=num_no_grad_iter,
                    info=info,
                    verbose=verbose,
                    truncated_grad_loop=False,
                    end_iter_callback=end_iter_callback,
                    **kwargs,
                )

            grad_loop_info = self._init_info(
                track_best_solution, track_err_history, track_state_history
            )
            grad_iters_done = self._optimize_loop(
                num_iter=backward_num_iterations,
                info=grad_loop_info,
                verbose=verbose,
                truncated_grad_loop=True,
                end_iter_callback=end_iter_callback,
                **kwargs,
            )

            # Adds grad_loop_info results to original info
            self._merge_infos(grad_loop_info, no_grad_iters_done, grad_iters_done, info)

            return info
        else:
            raise ValueError("Unrecognized backward mode")

    @abc.abstractmethod
    def compute_delta(self, **kwargs) -> torch.Tensor:
        pass

    # Resets any internal state needed by the optimizer for a new optimization
    # problem. Optimizer loop will pass all optimizer kwargs to this method.
    # Deliberately not abstract, since some optimizers might not need this
    def reset(self, **kwargs) -> None:
        pass

    # Called at the end of every optimizer step to update any internal state
    # of the optimizer
    @torch.no_grad()
    def update_optimizer_state(
        self,
        last_err: torch.Tensor,
        new_err: torch.Tensor,
        delta: torch.Tensor,
        **kwargs,
    ) -> None:
        self._update_state_impl(last_err, new_err, delta, **kwargs)

    # Deliberately not abstract, since some optimizers might not need this
    def _update_state_impl(
        self,
        last_err: torch.Tensor,
        new_err: torch.Tensor,
        delta: torch.Tensor,
        **kwargs,
    ) -> None:
        pass
