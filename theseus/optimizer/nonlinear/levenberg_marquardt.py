# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization
from theseus.optimizer.linear import (
    BaspachoSparseSolver,
    CholeskyDenseSolver,
    LinearSolver,
    LUCudaSparseSolver,
    LUDenseSolver,
)

from .nonlinear_least_squares import NonlinearLeastSquares

_LM_ALLOWED_ELLIPS_DAMP_SOLVERS: List[Type[LinearSolver]] = [
    BaspachoSparseSolver,
    CholeskyDenseSolver,
    LUDenseSolver,
    LUCudaSparseSolver,
]
_EDAMP_SOLVERS_STR = ",".join(c.__name__ for c in _LM_ALLOWED_ELLIPS_DAMP_SOLVERS)
_LM_ALLOWED_ADAPTIVE_DAMP_SOLVERS: List[Type[LinearSolver]] = [
    BaspachoSparseSolver,
    CholeskyDenseSolver,
    LUCudaSparseSolver,
    LUDenseSolver,
]
_ADAPT_DAMP_SOLVERS_STR = ",".join(
    c.__name__ for c in _LM_ALLOWED_ADAPTIVE_DAMP_SOLVERS
)


def _check_linear_solver(
    linear_solver: LinearSolver, allowed_solvers: Sequence[Type[LinearSolver]]
):
    good = False
    for lsc in allowed_solvers:
        if isinstance(linear_solver, lsc):
            good = True
    return good


# See Nocedal and Wright, Numerical Optimization, pp. 258 - 261
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
class LevenbergMarquardt(NonlinearLeastSquares):
    _MIN_DAMPING = 1.0e-7
    _MAX_DAMPING = 1.0e7

    def __init__(
        self,
        objective: Objective,
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
            linearization_cls=linearization_cls,
            linearization_kwargs=linearization_kwargs,
            linear_solver_kwargs=linear_solver_kwargs,
            abs_err_tolerance=abs_err_tolerance,
            rel_err_tolerance=rel_err_tolerance,
            max_iterations=max_iterations,
            step_size=step_size,
            **kwargs,
        )
        self._allows_ellipsoidal = _check_linear_solver(
            self.linear_solver, _LM_ALLOWED_ELLIPS_DAMP_SOLVERS
        )
        self._allows_adaptive = _check_linear_solver(
            self.linear_solver, _LM_ALLOWED_ADAPTIVE_DAMP_SOLVERS
        )
        self._damping: Union[float, torch.Tensor] = 0.001

    def reset(
        self,
        damping: float = 1e-3,
        adaptive_damping: bool = False,
        **kwargs,
    ) -> None:
        super().reset(**kwargs)
        if adaptive_damping and not self._allows_adaptive:
            raise NotImplementedError(
                f"Adaptive damping is only supported by solvers with type "
                f"[{_ADAPT_DAMP_SOLVERS_STR}], but got solver of type "
                f"{type(self.linear_solver).__name__}."
            )
        if adaptive_damping:
            self._damping = damping * torch.ones(
                self.objective.batch_size,
                device=self.objective.device,
                dtype=self.objective.dtype,
            )
        else:
            self._damping = damping

    # This method uses self._damping rather than the one in kwargs,
    # so that we can handle the adaptive case, which has memory
    def compute_delta(
        self,
        ellipsoidal_damping: bool = False,
        damping_eps: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if ellipsoidal_damping:
            if not self._allows_ellipsoidal:
                raise NotImplementedError(
                    f"Ellipsoidal damping is only supported by solvers with type "
                    f"[{_EDAMP_SOLVERS_STR}]."
                )
        if damping_eps is not None and not self._allows_ellipsoidal:
            raise NotImplementedError(
                f"damping eps is only supported by solvers with type "
                f"[{_EDAMP_SOLVERS_STR}]."
            )
        damping_eps = damping_eps if damping_eps is not None else 1e-8

        return self.linear_solver.solve(
            damping=self._damping,
            ellipsoidal_damping=ellipsoidal_damping,
            damping_eps=damping_eps,
        )

    def _complete_step(
        self,
        delta: torch.Tensor,
        new_err: torch.Tensor,
        previous_err: torch.Tensor,
        adaptive_damping: bool = False,
        down_damping_ratio: float = 9.0,
        up_damping_ratio: float = 11.0,
        damping_accept: float = 0.1,
        ellipsoidal_damping: bool = False,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        # "err" tensors passed as input refer to the squared norm of the
        # error vector, as returned by self.objective.error_metric()
        if adaptive_damping:
            return self._check_accept(
                delta,
                new_err,
                previous_err,
                damping_accept,
                down_damping_ratio,
                up_damping_ratio,
                ellipsoidal_damping,
            )
        else:
            return None

    # Checks if the step should be accepted (per batch element)
    # Adjusts self._damping accordingly
    # Returns a mask indicating which batch indices were accepted
    #
    # Based on https://people.duke.edu/~hpgavin/ce281/lm.pdf, Section 4.1
    # We currently use method (1) from 4.1.1
    @torch.no_grad()
    def _check_accept(
        self,
        delta: torch.Tensor,
        err: torch.Tensor,
        previous_err: torch.Tensor,
        damping_accept: float,
        down_damping_ratio: float,
        up_damping_ratio: float,
        ellipsoidal_damping: bool,
    ) -> torch.Tensor:
        linearization = self.linear_solver.linearization
        assert isinstance(self._damping, torch.Tensor)
        damping = self._damping.view(-1, 1)
        if ellipsoidal_damping:
            damping = linearization.diagonal_scaling(damping)
        # Deliberately using Atb before updating the variables, according to
        # the LM reference above
        den = (delta * (damping * delta + linearization.Atb.squeeze(2))).sum(dim=1) / 2
        rho = (previous_err - err) / den
        reject_indices = rho <= damping_accept
        self._damping = torch.where(
            reject_indices,
            self._damping * up_damping_ratio,
            self._damping / down_damping_ratio,
        )
        self._damping = self._damping.clamp(
            LevenbergMarquardt._MIN_DAMPING, LevenbergMarquardt._MAX_DAMPING
        )
        return reject_indices
