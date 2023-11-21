# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from theseus import CostFunction, CostWeight
from theseus.geometry import LieGroup
from theseus.global_params import _THESEUS_GLOBAL_PARAMS


class Local(CostFunction):
    def __init__(
        self,
        var: LieGroup,
        target: LieGroup,
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(var, target.__class__):
            raise ValueError(
                "Variable for the Local inconsistent with the given target."
            )
        if not var.dof() == target.dof():
            raise ValueError(
                "Variable and target in the Local must have identical dof."
            )
        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

        self._jac_cache: torch.Tensor = None

    def error(self) -> torch.Tensor:
        return self.target.local(self.var)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if _THESEUS_GLOBAL_PARAMS.fast_approx_local_jacobians:
            if (
                self._jac_cache is not None
                and self._jac_cache.shape[0] == self.var.shape[0]
            ):
                jacobian = self._jac_cache
            else:
                jacobian = torch.eye(
                    self.dim(), device=self.var.device, dtype=self.var.dtype
                ).repeat(self.var.shape[0], 1, 1)
                self._jac_cache = jacobian
            return (
                [jacobian],
                self.target.local(self.var),
            )
        else:
            Jlist: List[torch.Tensor] = []
            error = self.target.local(self.var, jacobians=Jlist)
            return [Jlist[1]], error

    def dim(self) -> int:
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "Local":
        return Local(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )
