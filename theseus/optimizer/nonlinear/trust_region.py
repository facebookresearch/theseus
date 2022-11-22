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

# This optimizer optimize() receives the following keywords (see p. 69):
#       - trust_region_init (default 1.0)
#       - accept_threshold (default 0.25)
#       - shrink_threshold (default 0.25)
#       - expand_threshold (default 0.75)
#       - shrink_ratio (default 0.25)
#       - expand_ratio (default 2.0)
#
# Iff the ratio between the actual error reduction after the step and the
# predicted one is greater than `accept_threshold`, the step is applied.
# Iff it's also greater than `expand_threshold`, the trust region is also
# increased by `expand_ratio`.
# Iff the ratio is lower than `shrink_threshold`,
# then the trust region is reduced by `shrink_ratio`.
# The trust region is initialized to `trust_region_init`.
class TrustRegionOptimizer(NonlinearLeastSquares, abc.ABC):
    _MIN_TRUST_REGION = 1.0e-7
    _MAX_TRUST_REGION = 1.0e7

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
        if not isinstance(self.linear_solver.linearization, DenseLinearization):
            # Since I will implement for sparse soon after,
            # I'll avoid fancier error handling
            # I expect this method to work with all our current solvers
            raise NotImplementedError
        self._trust_region: torch.Tensor = None
        self._trusted_step_idx: torch.Tensor = None

    def reset(
        self,
        trust_region_init: float = 1.0,
        **kwargs,
    ) -> None:
        self._trust_region = trust_region_init * torch.ones(
            self.objective.batch_size,
            1,
            device=self.objective.device,
            dtype=self.objective.dtype,
        )
        self._trusted_step_idx = None

    # Return the computed delta and, optionally, the indices that with the
    # steps that are within the trust region
    @abc.abstractmethod
    def _compute_delta_impl(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pass

    def compute_delta(self, **kwargs) -> torch.Tensor:
        # Storing the indices at the trust boundary so that we can update the
        # trust region inside self.complete_step()
        delta, self._trusted_step_idx = self._compute_delta_impl()
        if self._trusted_step_idx is None:
            self._trusted_step_idx = (
                TrustRegionOptimizer._detached_squared_norm(delta)
                <= self._trust_region**2
            )
        return delta

    @staticmethod
    @torch.no_grad()
    def _detached_squared_norm(
        tensor: torch.Tensor, keepdim: bool = True
    ) -> torch.Tensor:
        return (tensor**2).sum(dim=1, keepdim=keepdim)

    # Computes m_k(delta) as defined in Eq. (4.2) of the above reference (p. 68)
    @torch.no_grad()
    def _predicted_error(
        self, previous_error: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        linearization = self.linear_solver.linearization

        # TODO: For sparse I'll add a method like linearization.Avp() for Jacobian
        # vector product. The issue is that for the sparse linearization we need
        # to make this differentiable, because we are not using torch's
        # native functions
        assert isinstance(linearization, DenseLinearization)

        # Note that B = |A^t@ A| so that p^tBp := delta^t A^t A delta = ||Adelta||^2
        Adelta = linearization.A.bmm(delta.unsqueeze(2)).squeeze(2)
        grad = -linearization.Atb.squeeze(2)
        delta_dot_grad = (delta * grad).sum(dim=1)

        return (
            previous_error
            + delta_dot_grad
            + 0.5 * TrustRegionOptimizer._detached_squared_norm(Adelta, keepdim=False)
        )

    @torch.no_grad()
    def _complete_step(
        self,
        delta: torch.Tensor,
        new_err: torch.Tensor,
        previous_err: torch.Tensor,
        accept_threshold: float = 0.25,
        shrink_threshold: float = 0.25,
        expand_threshold: float = 0.75,
        shrink_ratio: float = 0.25,
        expand_ratio: float = 2.0,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        good_params = (0.0 < shrink_ratio <= 1.0) and (expand_ratio >= 1.0)
        good_params &= (shrink_threshold < expand_threshold) and (
            accept_threshold < expand_threshold
        )
        if not good_params:
            raise ValueError(
                "Invalid parameters for TrustRegionMethod. "
                "Values must satisfy accept_threshold/shrink_threshold < expand_threshold, "
                "shrink_ratio in (0, 1], and expand_ratio > 1.0."
            )
        pred_err = self._predicted_error(previous_err, delta)
        rho = ((previous_err - new_err) / (previous_err - pred_err)).view(-1, 1)
        shrink_idx = rho < shrink_threshold
        self._trust_region = torch.where(
            shrink_idx, self._trust_region * shrink_ratio, self._trust_region
        )
        expand_idx = (rho > expand_threshold) & ~self._trusted_step_idx
        self._trust_region = torch.where(
            expand_idx, self._trust_region * expand_ratio, self._trust_region
        )
        self._trust_region = self._trust_region.clamp(
            self._MIN_TRUST_REGION, self._MAX_TRUST_REGION
        )
        return (rho < accept_threshold).view(-1)
