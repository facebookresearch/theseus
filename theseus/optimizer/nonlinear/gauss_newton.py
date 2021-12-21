# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Type

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization
from theseus.optimizer.linear import LinearSolver

from .nonlinear_least_squares import NonlinearLeastSquares


class GaussNewton(NonlinearLeastSquares):
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

    def compute_delta(self, **kwargs) -> torch.Tensor:
        return self.linear_solver.solve()

    def compute_samples(self, n_samples: int = 10, T: float = 1.0) -> torch.Tensor:
        delta = self.linear_solver.solve()
        AtA = self.linear_solver.precision() / T
        sqrt_AtA = torch.linalg.cholesky(AtA).permute(0, 2, 1)

        batch_size, n_vars = delta.shape
        y = torch.normal(
            mean=torch.zeros((n_vars, n_samples), device=delta.device),
            std=torch.ones((n_vars, n_samples), device=delta.device),
        )
        delta_samples = (torch.triangular_solve(y, sqrt_AtA).solution) + (
            delta.unsqueeze(-1)
        ).repeat(1, 1, n_samples)

        x_samples = torch.zeros((batch_size, n_vars, n_samples), device=delta.device)
        for sidx in range(0, n_samples):
            var_idx = 0
            for var in self.linear_solver.linearization.ordering:
                new_var = var.retract(
                    delta_samples[:, var_idx : var_idx + var.dof(), sidx]
                )
                x_samples[:, var_idx : var_idx + var.dof(), sidx] = new_var.data
                var_idx = var_idx + var.dof()

        return x_samples
