# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple, Type

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
        vectorize: bool = False,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        linear_solver_kwargs: Optional[Dict[str, Any]] = None,
        abs_err_tolerance: float = 1e-10,
        rel_err_tolerance: float = 1e-8,
        max_iterations: int = 20,
        step_size: float = 1.0,
        **kwargs
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
            **kwargs
        )

    def compute_delta(self, **kwargs) -> torch.Tensor:
        return self.linear_solver.solve()

    def _step_impl(
        self,
        delta: torch.Tensor,
        steps: torch.Tensor,
        converged_indices: torch.Tensor,
        force_update: bool,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        self.objective.retract_optim_vars(
            delta * steps,
            self._tmp_optim_vars,
            ignore_mask=converged_indices,
            force_update=force_update,
        )
        tensor_map = {v.name: v.tensor for v in self._tmp_optim_vars}
        with torch.no_grad():
            error = self.objective.error_squared_norm(tensor_map, also_update=False)
        return tensor_map, error
