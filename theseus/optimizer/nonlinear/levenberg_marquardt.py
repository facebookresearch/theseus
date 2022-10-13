# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization
from theseus.optimizer.linear import (
    CholeskyDenseSolver,
    LinearSolver,
    LUCudaSparseSolver,
    LUDenseSolver,
)

from .nonlinear_least_squares import NonlinearLeastSquares

_LM_ALLOWED_ELLIPS_DAMP_SOLVERS: List[Type[LinearSolver]] = [
    CholeskyDenseSolver,
    LUDenseSolver,
    LUCudaSparseSolver,
]
_EDAMP_SOLVERS_STR = ",".join(c.__name__ for c in _LM_ALLOWED_ELLIPS_DAMP_SOLVERS)
_LM_ALLOWED_ADAPTIVE_DAMP_SOLVERS: List[Type[LinearSolver]] = [
    CholeskyDenseSolver,
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
    def __init__(
        self,
        objective: Objective,
        linear_solver_cls: Optional[Type[LinearSolver]] = None,
        vectorize: bool = True,
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
        self._ellipsoidal_damping = False
        self._adaptive_damping = False

    def reset(
        self,
        damping: float = 1e-3,
        ellipsoidal_damping: bool = False,
        adaptive_damping: bool = False,
        **kwargs,
    ) -> None:
        if ellipsoidal_damping and not self._allows_ellipsoidal:
            raise NotImplementedError(
                f"Ellipsoidal damping is only supported by solvers with type "
                f"[{_EDAMP_SOLVERS_STR}]."
            )
        self._ellipsoidal_damping = ellipsoidal_damping
        if adaptive_damping and not self._allows_adaptive:
            raise NotImplementedError(
                f"Adaptive damping is only supported by solvers with type "
                f"[{_ADAPT_DAMP_SOLVERS_STR}]."
            )
        if adaptive_damping:
            self._damping = damping * torch.ones(
                self.objective.batch_size,
                device=self.objective.device,
                dtype=self.objective.dtype,
            )
        else:
            self._damping = damping

    def compute_delta(
        self,
        damping_eps: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if damping_eps is not None and not self._allows_ellipsoidal:
            raise NotImplementedError(
                f"damping eps is only supported by solvers with type "
                f"[{_EDAMP_SOLVERS_STR}]."
            )
        damping_eps = damping_eps if damping_eps is not None else 1e-8

        return self.linear_solver.solve(
            damping=self._damping,
            ellipsoidal_damping=self._ellipsoidal_damping,
            damping_eps=damping_eps,
        )
