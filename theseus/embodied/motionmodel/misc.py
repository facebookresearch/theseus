# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch

from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.geometry import SE2, Point3, Vector


class HingeLoss(CostFunction):
    def __init__(
        self,
        vector: Vector,
        limit: Union[float, torch.Tensor, Variable],
        threshold: Union[float, torch.Tensor, Variable],
        cost_weight: CostWeight,
        name: Optional[str] = None,
        side: str = "both",
    ):
        super().__init__(cost_weight, name=name)
        self.vector = vector
        self.limit = as_variable(limit, name=f"{self.name}__vlimit")
        self.threshold = as_variable(threshold, name=f"{self.name}__vthres")
        for v in [self.limit, self.threshold]:
            if not v.ndim == 2 and v.shape[1] == 1:
                raise ValueError("Limit and threshold must be scalar variables.")
        self.register_optim_var("vector")
        self.register_aux_vars(["limit", "threshold"])
        if side not in ["below", "above", "both"]:
            raise ValueError("side must be one of 'both', 'above', 'below'.")
        self.side = side

    def dim(self):
        return self.vector.dof()

    def _compute_error(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vector = self.vector.tensor
        limit = self.limit.tensor
        threshold = self.threshold.tensor
        down_limit = -limit + threshold
        up_limit = limit - threshold
        below_idx = vector < down_limit
        above_idx = vector > up_limit
        error = vector.new_zeros(vector.shape)
        if self.side in ["below", "both"]:
            error = torch.where(below_idx, down_limit - vector, error)
        if self.side in ["above", "both"]:
            error = torch.where(above_idx, vector - up_limit, error)
        return error, below_idx, above_idx

    def error(self) -> torch.Tensor:
        return self._compute_error()[0]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error, below_idx, above_idx = self._compute_error()
        J = self.vector.tensor.new_zeros(
            self.vector.shape[0], self.dim(), self.vector.dof()
        )
        if self.side in ["below", "both"]:
            J[below_idx.diag_embed()] = -1.0
        if self.side in ["above", "both"]:
            J[above_idx.diag_embed()] = 1.0
        return [J], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "HingeLoss":
        return HingeLoss(
            self.vector.copy(),
            self.limit.copy(),
            self.threshold.copy(),
            self.weight.copy(),
            name=new_name,
            side=self.side,
        )


class Nonholonomic(CostFunction):
    def __init__(
        self,
        pose: Union[SE2, Point3, Vector],
        vel: Union[Point3, Vector],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        if vel.dof() != 3 or pose.dof() != 3:
            raise ValueError(
                "Nonholonomic only accepts 3D velocity or poses (x, y, z dims). "
                "Poses can either be SE2 or Vector variables. Velocities only Vector."
            )
        self.pose = pose
        self.vel = vel
        self.register_optim_vars(["pose", "vel"])
        self.weight = cost_weight

    def dim(self):
        return 1

    def _compute_error(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        vel = self.vel.tensor
        if isinstance(self.pose, SE2):
            error = vel[:, 1]
            cos, sin = None, None
        else:
            cos = self.pose[:, 2].cos()
            sin = self.pose[:, 2].sin()
            error = vel[:, 1] * cos - vel[:, 0] * sin
        return error.view(-1, 1), cos, sin

    def error(self) -> torch.Tensor:
        return self._compute_error()[0]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Pre-allocate jacobian tensors
        batch_size = self.vel.shape[0]
        Jvel = self.vel.tensor.new_zeros(batch_size, 1, 3)
        error, cos, sin = self._compute_error()
        if isinstance(self.pose, SE2):
            Jpose = error.new_zeros(error.shape[0], 1, 3)
            Jvel = error.new_zeros(error.shape[0], 1, 3)
            Jvel[:, 0, 1] = 1
        else:
            Jpose = self.vel.tensor.new_zeros(self.vel.shape[0], 1, 3)
            Jvel = self.vel.tensor.new_zeros(self.vel.shape[0], 1, 3)
            Jpose[:, 0, 2] = -(self.vel[:, 1] * sin + self.vel[:, 0] * cos)
            Jvel[:, 0, 0] = -sin
            Jvel[:, 0, 1] = cos
        return [Jpose, Jvel], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "Nonholonomic":
        return Nonholonomic(
            self.pose.copy(),
            self.vel.copy(),
            self.weight.copy(),
            name=new_name,
        )
