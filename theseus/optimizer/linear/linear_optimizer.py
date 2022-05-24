import warnings
from enum import Enum
from typing import Any, Dict, Optional, Type

import numpy as np
import torch

from theseus.core import Objective
from theseus.optimizer import Linearization, Optimizer, OptimizerInfo

from .linear_solver import LinearSolver


class LinearOptimizerStatus(Enum):
    START = 0
    CONVERGED = 1
    FAIL = -1


class LinearOptimizer(Optimizer):
    def __init__(
        self,
        objective: Objective,
        linear_solver_cls: Type[LinearSolver],
        *args,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        linear_solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(objective)
        linearization_kwargs = linearization_kwargs or {}
        linear_solver_kwargs = linear_solver_kwargs or {}
        self.linear_solver = linear_solver_cls(
            objective,
            linearization_cls=linearization_cls,
            linearization_kwargs=linearization_kwargs,
            **linear_solver_kwargs,
        )

    def _optimize_impl(
        self,
        **kwargs,
    ) -> OptimizerInfo:

        info = OptimizerInfo(
            best_solution={},
            status=np.array([LinearOptimizerStatus.START] * self.objective.batch_size),
        )
        try:
            self.linear_solver.linearization.linearize()
            delta = self.linear_solver.solve()
        except RuntimeError as run_err:
            msg = (
                f"There was an error while running the linear optimizer. "
                f"Original error message: {run_err}."
            )
            if torch.is_grad_enabled():
                raise RuntimeError(
                    msg + " Backward pass will not work. To obtain "
                    "the best solution seen before the error, run with torch.no_grad()"
                )
            else:
                warnings.warn(msg, RuntimeWarning)
                info.status[:] = LinearOptimizerStatus.FAIL
                return info
        self.retract_and_update_variables(delta)
        info.status[:] = LinearOptimizerStatus.CONVERGED
        for var in self.linear_solver.linearization.ordering:
            info.best_solution[var.name] = var.data.clone().cpu()
        return info

    # retracts all variables in the given order and updates their values
    # with the result
    def retract_and_update_variables(self, delta: torch.Tensor):
        var_idx = 0
        for var in self.linear_solver.linearization.ordering:
            new_var = var.retract(delta[:, var_idx : var_idx + var.dof()])
            var.update(new_var.data)
            var_idx += var.dof()
