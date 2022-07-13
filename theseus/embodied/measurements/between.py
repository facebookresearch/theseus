# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from theseus.core import CostFunction, CostWeight
from theseus.geometry import LieGroup, between


class Between(CostFunction):
    def __init__(
        self,
        v0: LieGroup,
        v1: LieGroup,
        measurement: LieGroup,
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.v0 = v0
        self.v1 = v1
        self.register_optim_vars(["v0", "v1"])
        self.measurement = measurement
        self.register_aux_vars(["measurement"])
        if not isinstance(v0, v1.__class__) or not isinstance(
            v0, measurement.__class__
        ):
            raise ValueError("Inconsistent types between variables and measurement.")

    def error(self) -> torch.Tensor:
        var_diff = between(self.v0, self.v1)
        return self.measurement.local(var_diff)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        Jlist: List[torch.Tensor] = []
        var_diff = between(self.v0, self.v1)
        log_jac: List[torch.Tensor] = []
        error = self.measurement.inverse().compose(var_diff).log_map(jacobians=log_jac)
        dlog = log_jac[0]
        Jlist.extend([-dlog @ var_diff.inverse().adjoint(), dlog])
        return Jlist, error

    def dim(self) -> int:
        return self.v0.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "Between":
        return Between(
            self.v0.copy(),
            self.v1.copy(),
            self.measurement.copy(),
            self.weight.copy(),
            name=new_name,
        )
