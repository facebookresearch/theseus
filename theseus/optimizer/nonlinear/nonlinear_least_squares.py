# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Type, Union

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization, OptimizerInfo
from theseus.optimizer.linear import (
    LinearSolver,
    CholeskyDenseSolver,
    LUCudaSparseSolver,
)
from .nonlinear_optimizer import (
    BackwardMode,
    NonlinearOptimizer,
    NonlinearOptimizerInfo,
    NonlinearOptimizerStatus,
)


EndIterCallbackType = Callable[
    ["NonlinearOptimizer", NonlinearOptimizerInfo, torch.Tensor, int], NoReturn
]


# Base class for all optimizers for NLLS problems,
# providing the skeleton of the
# optimization loop. Subclasses need to implement the following method:
#
#   - `compute_delta`: returns a descent direction given the current values
#     of the objective's optimization vars.
#
# Optionally, they can also provide the following methods:
#
#   - `reset`: resets any internal state needed by the optimizer.
#   - `_complete_step`: called at the end of an optimization step, but before
#     optimization variables are updated. Returns batch indices that should not
#     be updated (e.g., if the step is to be rejected).
#
# The high level logic of a call to optimize is as follows:
#
# prev_err = objective.error_squared_norm() / 2
# do optimization loop:
#    1. compute delta
#    2. step(delta, prev_err)
#           2.1. Store current optim var tensors in tmp_optim_vars containers
#           2.2. Retract all tmp_optim_vars given delta
#           2.3. Evaluate new error
#           2.4. reject_indices = self._complete_step(delta, new_err, prev_err)
#           2.5. Update objective's optim var containers with retracted values,
#                ignoring indices given by `reject_indices`
#    3. Check convergence
class NonlinearLeastSquares(NonlinearOptimizer, abc.ABC):
    def __init__(
        self,
        objective: Objective,
        *args,
        linear_solver_cls: Optional[Type[LinearSolver]] = None,
        vectorize: bool = False,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        linear_solver_kwargs: Optional[Dict[str, Any]] = None,
        abs_err_tolerance: float = 1e-10,
        rel_err_tolerance: float = 1e-8,
        max_iterations: int = 20,
        step_size: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            objective,
            linear_solver_cls=linear_solver_cls,
            vectorize=vectorize,
            abs_err_tolerance=abs_err_tolerance,
            rel_err_tolerance=rel_err_tolerance,
            max_iterations=max_iterations,
            step_size=step_size,
            **kwargs,
        )
        linear_solver_cls = linear_solver_cls or CholeskyDenseSolver
        linear_solver_kwargs = linear_solver_kwargs or {}
        self.linear_solver = linear_solver_cls(
            objective,
            linearization_cls=linearization_cls,
            linearization_kwargs=linearization_kwargs,
            **linear_solver_kwargs,
        )
        self.ordering = self.linear_solver.linearization.ordering
        self._tmp_optim_vars = tuple(v.copy(new_name=v.name) for v in self.ordering)

    def _error_metric(
        self,
        input_tensors: Optional[Dict[str, torch.Tensor]] = None,
        also_update: bool = False,
    ) -> torch.Tensor:
        return (
            self.objective.error_squared_norm(
                input_tensors=input_tensors, also_update=also_update
            )
            / 2
        )

    # loop for the iterative optimizer
    def _optimize_loop(
        self,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        _last_implicit_diff_step: bool = False,
        **kwargs,
    ) -> int:
        steps_tensor: torch.Tensor = None  # type: ignore
        converged_indices = torch.zeros_like(info.last_err).bool()
        iters_done = 0
        it_ = 0
        all_reject_attempts = 0
        while it_ < num_iter:
            # do optimizer step
            # See comment inside `if _last_implicit_diff_step` case below
            self.linear_solver.linearization.linearize(
                _detach_hessian=_last_implicit_diff_step,
            )
            try:
                if _last_implicit_diff_step:
                    # The derivation for implicit differentiation states that
                    # the autograd-enabled loop must be done using Gauss-Newton steps.
                    # Well, technically full Newton, this is hard to implement and GN
                    # is working well so far.
                    #
                    # We also need to detach the hessian when computing
                    # linearization above, as higher order terms introduce errors
                    # in the derivative if the fixed point is not accurate enough.
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

            if _last_implicit_diff_step:
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

            # For now, step size is combined with delta. If we add more sophisticated
            # line search, will probably need to pass it separately, or compute inside.
            err, all_rejected = self._step(
                delta * steps_tensor,
                info.last_err,
                converged_indices,
                force_update,
                delta_forced_to_gn=_last_implicit_diff_step,
                **kwargs,
            )  # err is shape (batch_size,)
            if all_rejected:
                # The optimizer rejected all steps so just continue, otherwise convergence
                # check will stop prematurely. However, we put a max on this to guarantee
                # this terminates
                all_reject_attempts += 1
                if all_reject_attempts < NonlinearOptimizer._MAX_ALL_REJECT_ATTEMPTS:
                    continue
            all_reject_attempts = 0

            # check for convergence if at least one step was accepted
            with torch.no_grad():
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

                if end_iter_callback is not None:
                    end_iter_callback(self, info, delta, it_)

            iters_done += 1
            it_ += 1

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
        backward_mode = BackwardMode.resolve(backward_mode)
        kwargs_plus_bwd_mode = {**kwargs, **{"backward_mode": backward_mode}}
        self.reset(**kwargs_plus_bwd_mode)
        with torch.no_grad():
            info = self._init_info(
                track_best_solution, track_err_history, track_state_history
            )

        if verbose:
            print(
                f"Nonlinear optimizer. Iteration: 0. "
                f"Error: {info.last_err.mean().item()}"
            )

        backward_num_iters, no_grad_num_iters = self._split_backward_iters(
            **kwargs_plus_bwd_mode
        )
        if backward_mode in [BackwardMode.UNROLL, BackwardMode.DLM]:
            self._optimize_loop(
                start_iter=0,
                num_iter=backward_num_iters,
                info=info,
                verbose=verbose,
                end_iter_callback=end_iter_callback,
                _last_implicit_diff_step=False,
                **kwargs,
            )
            # If didn't coverge, remove misleading converged_iter value
            info.converged_iter[
                info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
            ] = -1
            return info
        elif backward_mode in [BackwardMode.IMPLICIT, BackwardMode.TRUNCATED]:
            with torch.no_grad():
                # actual_num_iters could be < num_iter due to early convergence
                no_grad_iters_done = self._optimize_loop(
                    num_iter=no_grad_num_iters,
                    info=info,
                    verbose=verbose,
                    end_iter_callback=end_iter_callback,
                    _last_implicit_diff_step=False,
                    **kwargs,
                )

            grad_loop_info = self._init_info(
                track_best_solution, track_err_history, track_state_history
            )
            grad_iters_done = self._optimize_loop(
                num_iter=backward_num_iters,
                info=grad_loop_info,
                verbose=verbose,
                end_iter_callback=end_iter_callback,
                _last_implicit_diff_step=backward_mode == BackwardMode.IMPLICIT,
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

    # Adds references to the current optim variable tensors in the the optimizer's
    # _tmp_optim_varscontainers. This allow us to compute t_next = V.tensor + delta for
    # any optimization variable, without changing the permanent optim var objects
    # in the objective.
    def _update_tmp_optim_vars(self):
        for v_tmp, v_order in zip(self._tmp_optim_vars, self.ordering):
            v_tmp.update(v_order.tensor)

    def _compute_retracted_tensors_and_error(
        self,
        delta: torch.Tensor,
        converged_indices: torch.Tensor,
        force_update: bool,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # makes sure tmp containers are up to date with current variables
        self._update_tmp_optim_vars()
        # stores the result of the retract step in `self._tmp_optim_vars`
        self.objective.retract_vars_sequence(
            delta,
            self._tmp_optim_vars,
            ignore_mask=converged_indices,
            force_update=force_update,
        )
        tensor_dict = {v.name: v.tensor for v in self._tmp_optim_vars}
        err = self._error_metric(tensor_dict, also_update=False)
        return tensor_dict, err

    # Given descent directions and step sizes, updates the optimization
    # variables.
    # Batch indices indicated by `converged_indices` mask are ignored
    # unless `force_update = True`.
    # Returns the total error tensor after the update and a boolean indicating
    # if all steps were rejected.
    #
    # `previous_err` refes to the squared norm of the
    # error vector, as returned by self._error_metric()
    # The returned error must also refer to this quantity, but after the
    # update.
    def _step(
        self,
        delta: torch.Tensor,
        previous_err: torch.Tensor,
        converged_indices: torch.Tensor,
        force_update: bool,
        delta_forced_to_gn: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, bool]:
        tensor_dict, err = self._compute_retracted_tensors_and_error(
            delta, converged_indices, force_update
        )
        if delta_forced_to_gn:
            # If delta has been forced to be a GN step (for implicit diff),
            # then we need to make sure we ignore any reject indices computed by
            # other methods like LM or Dogleg (so the step is not rejected)
            reject_indices = None
        else:
            reject_indices = self._complete_step(delta, err, previous_err, **kwargs)

        if reject_indices is not None and reject_indices.all():
            return previous_err, True

        self.objective.update(tensor_dict, batch_ignore_mask=reject_indices)
        if reject_indices is not None and reject_indices.any():
            # Some steps were rejected so the error computed above is not accurate
            err = self._error_metric()
        return err, False

    # Resets any internal state needed by the optimizer for a new optimization
    # problem. Optimizer loop will pass all optimizer kwargs to this method.
    # Deliberately not abstract, since some optimizers might not need this
    def reset(self, **kwargs) -> None:
        backward_num_iters, _ = self._split_backward_iters(**kwargs)
        if (
            isinstance(self.linear_solver, LUCudaSparseSolver)
            and "num_solver_contexts" not in kwargs
        ):
            # Auto set number of solver context for the given max iterations
            kwargs["num_solver_contexts"] = (
                backward_num_iters * NonlinearOptimizer._MAX_ALL_REJECT_ATTEMPTS
            )
        self.linear_solver.reset(**kwargs)

    # Called at the end of step() but before variables are update to their new values.
    # This method can be used to update any internal state of the optimizer and
    # also obtain an optional tensor of shape (batch_size,), representing
    # a mask of booleans indicating if the step is to be rejected for any
    # batch elements.
    #
    # Deliberately not abstract, since some optimizers might not need this.
    def _complete_step(
        self,
        delta: torch.Tensor,
        new_err: torch.Tensor,
        previous_err: torch.Tensor,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        return None
