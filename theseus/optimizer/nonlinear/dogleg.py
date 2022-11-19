# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Optional, Tuple, Type

import torch

from theseus.core import Objective
from theseus.optimizer import DenseLinearization, Linearization
from theseus.optimizer.linear import LinearSolver

from .trust_region import TrustRegionOptimizer


# See Nocedal and Wright, Numerical Optimization, pp. 73-77
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
class Dogleg(TrustRegionOptimizer):
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

    def reset(
        self,
        trust_region_init: float = 1.2,
        **kwargs,
    ) -> None:
        self._trust_region = trust_region_init

    @staticmethod
    @torch.no_grad()
    def _detached_squared_norm(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor**2).sum(dim=1)

    def _compute_delta_impl(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        trust_region_2 = self._trust_region**2
        delta_gn = self.linear_solver.solve()
        delta_gn_norm_2 = Dogleg._detached_squared_norm(delta_gn)
        good_gn_idx = delta_gn_norm_2 < trust_region_2
        # All Gauss-Newton step are within trust-region, can return
        if good_gn_idx.all():
            return delta_gn, ~good_gn_idx  # no indices at boundary

        # If reach here, some steps are outside trust-region,
        # need to compute dogleg step

        linearization = self.linear_solver.linearization
        # TODO: For sparse I'll add a method like linearization.Avp() for Jacobian
        # vector product. The issue is that for the sparse linearization we need
        # to make this differentiable, because we are not using torch's
        # native functions
        assert isinstance(linearization, DenseLinearization)

        grad = linearization.Atb.squeeze(2)
        Adelta_sd = linearization.A.bmm(grad.unsqueeze(2)).squeeze(2)
        Adelta_sd_norm_2 = Dogleg._detached_squared_norm(Adelta_sd)
        grad_norm_2 = Dogleg._detached_squared_norm(grad)
        t = grad_norm_2 / Adelta_sd_norm_2
        delta_sd = grad * t.view(-1, 1)
        delta_sd_norm_2 = Dogleg._detached_squared_norm(delta_sd)

        delta_dogleg = delta_sd
        good_sd_idx = delta_sd_norm_2 <= trust_region_2
        if good_sd_idx.any():
            # In this case, some steepest descent steps are within region
            # so need to extend towards boundary with Gauss-Newton step
            # Need to solve a quadratic || sd + tau * (gn - sd)|| == tr**2
            # This can be written as
            # a * tau^2 + b * tau + c, with a, b, c given below
            diff = delta_gn - delta_sd
            with torch.no_grad():
                a = Dogleg._detached_squared_norm(diff)
                b = (2 * delta_sd * diff).sum(dim=1)
                c = delta_sd_norm_2 - trust_region_2
                tau = (-b + ((b**2) - 4 * a * c).sqrt()) / (2 * a)
            delta_dogleg = torch.where(
                good_sd_idx, delta_sd + tau.view(-1, 1) * diff, delta_sd
            )

        if (~good_sd_idx).any():
            # For any steps beyond the trust region, truncate to trust region
            delta_dogleg = torch.where(
                ~good_sd_idx,
                delta_sd * trust_region_2 / delta_sd_norm_2.sqrt(),
                delta_dogleg,
            )

        return delta_dogleg, ~good_gn_idx
