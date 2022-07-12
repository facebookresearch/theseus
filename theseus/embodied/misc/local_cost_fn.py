# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from theseus import CostFunction, CostWeight
from theseus.geometry import LieGroup


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

    def error(self) -> torch.Tensor:
        return self.target.local(self.var)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlist: List[torch.Tensor] = []
        self.target.local(self.var, jacobians=Jlist)
        return [Jlist[1]], self.error()

    def dim(self) -> int:
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "Local":
        return Local(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )
