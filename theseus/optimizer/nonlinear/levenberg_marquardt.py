# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Type

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization
from theseus.optimizer.linear import DenseSolver, LinearSolver, LUCudaSparseSolver

from .nonlinear_least_squares import NonlinearLeastSquares

_LM_ALLOWED_ELLIPS_DAMP_SOLVERS = [DenseSolver, LUCudaSparseSolver]


def _check_ellipsoidal_damping_cls(linear_solver: LinearSolver):
    good = False
    for lsc in _LM_ALLOWED_ELLIPS_DAMP_SOLVERS:
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
        self._allows_ellipsoidal = _check_ellipsoidal_damping_cls(self.linear_solver)

    def compute_delta(
        self,
        damping: float = 1e-3,
        ellipsoidal_damping: bool = False,
        damping_eps: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:

        solvers_str = ",".join(c.__name__ for c in _LM_ALLOWED_ELLIPS_DAMP_SOLVERS)
        if ellipsoidal_damping and not self._allows_ellipsoidal:
            raise NotImplementedError(
                f"Ellipsoidal damping is only supported by solvers with type "
                f"[{solvers_str}]."
            )
        if damping_eps and not self._allows_ellipsoidal:
            raise NotImplementedError(
                f"damping eps is only supported by solvers with type "
                f"[{solvers_str}]."
            )
        damping_eps = damping_eps or 1e-8

        return self.linear_solver.solve(
            damping=damping,
            ellipsoidal_damping=ellipsoidal_damping,
            damping_eps=damping_eps,
        )
