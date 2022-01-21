# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from theseus.optimizer import Optimizer, OptimizerInfo
from theseus.optimizer.linear import LinearSolver


class TheseusLayer(nn.Module):
    def __init__(
        self,
        optimizer: Optimizer,
    ):
        super().__init__()
        self.objective = optimizer.objective
        self.optimizer = optimizer
        self._objectives_version = optimizer.objective.current_version

    def forward(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], OptimizerInfo]:
        if self._objectives_version != self.objective.current_version:
            raise RuntimeError(
                "The objective was modified after the layer's construction, which is "
                "currently not supported."
            )
        self.objective.update(input_data)
        optimizer_kwargs = optimizer_kwargs or {}
        info = self.optimizer.optimize(**optimizer_kwargs)
        values = dict(
            [
                (var_name, var.data)
                for var_name, var in self.objective.optim_vars.items()
            ]
        )
        return values, info

    def compute_samples(
        self,
        linear_solver: LinearSolver = None,
        n_samples: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # When samples are not available, return None. This makes the outer learning loop default
        # to a perceptron loss using the mean trajectory solution from the optimizer.
        if linear_solver is None:
            return None

        # Sampling from multivariate normal using a Cholesky decomposition of AtA,
        # http://www.statsathome.com/2018/10/19/sampling-from-multivariate-normal-precision-and-covariance-parameterizations/
        delta = linear_solver.solve()
        AtA = linear_solver.linearization.hessian_approx() / temperature
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
            for var in linear_solver.linearization.ordering:
                new_var = var.retract(
                    delta_samples[:, var_idx : var_idx + var.dof(), sidx]
                )
                x_samples[:, var_idx : var_idx + var.dof(), sidx] = new_var.data
                var_idx = var_idx + var.dof()

        return x_samples

    # Applies to() with given args to all tensors in the objective
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.objective.to(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.objective.device

    @property
    def dtype(self) -> torch.dtype:
        return self.objective.dtype
