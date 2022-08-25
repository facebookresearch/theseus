# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from theseus import CostFunction, CostWeight
from theseus.geometry import LieGroup, Point2, SE2


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

    def _error_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return self.target.local(self.var, jacobians=jacobians)

    def error(self) -> torch.Tensor:
        return self._error_impl()

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlist: List[torch.Tensor] = []
        error = self._error_impl(jacobians=Jlist)
        return [Jlist[1]], error

    def dim(self) -> int:
        return self.var.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "Local":
        return Local(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )


class XYDifference(CostFunction):
    def __init__(
        self,
        var: SE2,
        target: Point2,
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if not isinstance(var, SE2) and not isinstance(target, Point2):
            raise ValueError(
                "XYDifference expects var of type SE2 and target of type Point2."
            )
        self.var = var
        self.target = target
        self.register_optim_vars(["var"])
        self.register_aux_vars(["target"])

    def _jacobians_and_error_impl(
        self, compute_jacobians: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlocal: List[torch.Tensor] = [] if compute_jacobians else None
        Jxy: List[torch.Tensor] = [] if compute_jacobians else None
        error = self.target.local(self.var.xy(jacobians=Jxy), jacobians=Jlocal)
        jac = [Jlocal[1].matmul(Jxy[0])] if compute_jacobians else None
        return jac, error

    def error(self) -> torch.Tensor:
        return self._jacobians_and_error_impl(compute_jacobians=False)[1]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self._jacobians_and_error_impl(compute_jacobians=True)

    def dim(self) -> int:
        return 2

    def _copy_impl(self, new_name: Optional[str] = None) -> "XYDifference":
        return XYDifference(  # type: ignore
            self.var.copy(), self.target.copy(), self.weight.copy(), name=new_name
        )
