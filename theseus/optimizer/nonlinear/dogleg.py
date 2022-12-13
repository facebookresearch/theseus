# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Optional, Type

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization
from theseus.optimizer.linear import LinearSolver

from .trust_region import TrustRegion


# See Nocedal and Wright, Numerical Optimization, pp. 73-77
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
class Dogleg(TrustRegion):
    EPS = 1e-7

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

    def _compute_delta_impl(self) -> torch.Tensor:
        trust_region_2 = self._trust_region**2
        delta_gn = self.linear_solver.solve()
        delta_gn_norm_2 = TrustRegion._squared_norm(delta_gn)
        # All Gauss-Newton step are within trust-region, can return
        if (delta_gn_norm_2 < trust_region_2).all():
            # Return a False mask since no indices are at the boundary
            return delta_gn

        # ---------------------------------------------------------------------
        # Some Gauss-Newton steps are outside trust-region,
        # need to compute dogleg step
        # ---------------------------------------------------------------------
        linearization = self.linear_solver.linearization

        delta_sd = linearization.Atb.squeeze(2)
        Adelta_sd = linearization.Av(delta_sd)
        Adelta_sd_norm_2 = TrustRegion._squared_norm(Adelta_sd)
        grad_norm_2 = TrustRegion._squared_norm(delta_sd)
        cauchy_step_size = grad_norm_2 / (Adelta_sd_norm_2 + Dogleg.EPS)
        delta_c = delta_sd * cauchy_step_size
        delta_c_norm_2 = grad_norm_2 * (cauchy_step_size**2)
        delta_c_within_region_idx = delta_c_norm_2 <= trust_region_2

        # First make sure that any steps beyond the trust region, are truncated
        if not delta_c_within_region_idx.all():
            delta_dogleg = torch.where(
                delta_c_within_region_idx,
                delta_c,
                delta_c * self._trust_region / (delta_c_norm_2 + Dogleg.EPS).sqrt(),
            )
        else:
            delta_dogleg = delta_c

        if delta_c_within_region_idx.any():
            # In this case, some Cauchy steps are within the trust region
            # so need to extend towards boundary with Gauss-Newton step
            # Need to solve a quadratic || sd + tau * (gn - sd)|| == tr**2
            # This can be written as
            # a * tau^2 + b * tau + c, with a, b, c given below
            diff = delta_gn - delta_c
            a = TrustRegion._squared_norm(diff)
            b = (2 * delta_c * diff).sum(dim=1, keepdim=True)
            c = delta_c_norm_2 - trust_region_2
            disc = ((b**2) - 4 * a * c).clamp(Dogleg.EPS)
            # By taking min(tau, 1), this also covers the case when ||d_gn|| < TR
            tau = ((-b + disc.sqrt()) / (2 * a + Dogleg.EPS)).clamp(max=1.0)
            delta_dogleg = torch.where(
                delta_c_within_region_idx,
                delta_c + tau * diff,
                delta_dogleg,
            )

        # The only steps that are within the trust region are those were
        # Gauss-Newton was "good". Every other step size will have
        # norm exactly equal to the trust region
        return delta_dogleg
