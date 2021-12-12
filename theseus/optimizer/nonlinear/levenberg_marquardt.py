# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Type

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization
from theseus.optimizer.linear import DenseSolver, LinearSolver

from .nonlinear_least_squares import NonlinearLeastSquares


# See Nocedal and Wright, Numerical Optimization, pp. 258 - 261
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
class LevenbergMarquardt(NonlinearLeastSquares):
    def __init__(
        self,
        objective: Objective,
        linear_solver_cls: Optional[Type[LinearSolver]] = None,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        linear_solver_kwargs: Optional[Dict[str, Any]] = None,
        abs_err_tolerance: float = 1e-10,
        rel_err_tolerance: float = 1e-8,
        max_iterations: int = 20,
        step_size: float = 1.0,
    ):
        super().__init__(
            objective,
            linear_solver_cls=linear_solver_cls,
            linearization_cls=linearization_cls,
            linearization_kwargs=linearization_kwargs,
            linear_solver_kwargs=linear_solver_kwargs,
            abs_err_tolerance=abs_err_tolerance,
            rel_err_tolerance=rel_err_tolerance,
            max_iterations=max_iterations,
            step_size=step_size,
        )

    def compute_delta(
        self,
        damping: float = 1e-3,
        ellipsoidal_damping: bool = False,
        damping_eps: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if ellipsoidal_damping:
            raise NotImplementedError("Ellipsoidal damping is not currently supported.")
        if ellipsoidal_damping and not isinstance(self.linear_solver, DenseSolver):
            raise NotImplementedError(
                "Ellipsoidal damping is only supported when using DenseSolver."
            )
        if damping_eps and not isinstance(self.linear_solver, DenseSolver):
            raise NotImplementedError(
                "damping eps is only supported when using DenseSolver."
            )
        damping_eps = damping_eps or 1e-8

        return self.linear_solver.solve(
            damping=damping,
            ellipsoidal_damping=ellipsoidal_damping,
            damping_eps=damping_eps,
        )

    def compute_samples(self, n_samples: int = 10, T: float = 1.0) -> torch.Tensor:
        delta = self.linear_solver.solve()
        AtA = self.linear_solver.precision() / T
        R = torch.linalg.cholesky(AtA, upper=True)

        B, N = delta.shape
        y = torch.normal(
            mean=torch.zeros((N, n_samples), device=delta.device),
            std=torch.ones((N, n_samples), device=delta.device),
        )
        delta_samples = (torch.triangular_solve(y, R).solution) + (
            delta.unsqueeze(-1)
        ).repeat(1, 1, n_samples)

        x_samples = torch.zeros((B, N, n_samples), device=delta.device)
        for sidx in range(0, n_samples):
            var_idx = 0
            for var in self.linear_solver.linearization.ordering:
                new_var = var.retract(
                    delta_samples[:, var_idx : var_idx + var.dof(), sidx]
                )
                x_samples[:, var_idx : var_idx + var.dof(), sidx] = new_var.data
                var_idx = var_idx + var.dof()

        return x_samples
