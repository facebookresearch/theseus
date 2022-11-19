# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Any, Dict, Optional, Tuple, Type

import torch

from theseus.core import Objective
from theseus.optimizer import DenseLinearization, Linearization
from theseus.optimizer.linear import LinearSolver

from .nonlinear_least_squares import NonlinearLeastSquares


# See Nocedal and Wright, Numerical Optimization, Chapter 4
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
class TrustRegionOptimizer(NonlinearLeastSquares, abc.ABC):
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
        if linearization_cls not in [DenseLinearization]:
            # Since I will implement for sparse soon after,
            # I'll avoid fancier error handling
            # I expect this method to work with all our current solvers
            raise NotImplementedError
        self._trust_region = 1.0
        self._at_trust_boundary = None

    def reset(
        self,
        trust_region_init: float = 1.0,
        **kwargs,
    ) -> None:
        self._trust_region = trust_region_init
        self._at_trust_boundary = None

    # Return the computed delta and optionally the indices that reached
    # the trust boundary of the trust region
    @abc.abstractmethod
    def _compute_delta_impl(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    @staticmethod
    @torch.no_grad()
    def _detached_squared_norm(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor**2).sum(dim=1)

    def compute_delta(self, **kwargs) -> torch.Tensor:
        # Storing the indices at the trust boundary so that we can update the
        # trust region inside self.complete_step()
        delta, self._at_trust_boundary_idx = self._compute_delta_impl()
        if self._at_trust_boundary_idx is None:
            self._at_trust_boundary_idx = (
                TrustRegionOptimizer._detached_squared_norm(delta)
                >= self._trust_region**2
            )
        return delta
