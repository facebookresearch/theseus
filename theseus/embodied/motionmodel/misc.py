# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch

from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.geometry import SE2, Point3, Vector


class HingeCost(CostFunction):
    def __init__(
        self,
        vector: Vector,
        down_limit: Union[float, torch.Tensor, Variable],
        up_limit: Union[float, torch.Tensor, Variable],
        threshold: Union[float, torch.Tensor, Variable],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.vector = vector
        down_limit = HingeCost._convert_to_tensor_if_float(down_limit, vector.dof())
        up_limit = HingeCost._convert_to_tensor_if_float(up_limit, vector.dof())
        threshold = HingeCost._convert_to_tensor_if_float(threshold, vector.dof())
        self.down_limit = as_variable(down_limit, name=f"{self.name}__downlimit")
        self.up_limit = as_variable(up_limit, name=f"{self.name}__uplimit")
        self.threshold = as_variable(threshold, name=f"{self.name}__thres")
        for v in [self.down_limit, self.up_limit, self.threshold]:
            if not v.ndim == 2 or not v.shape[1] == vector.dof():
                raise ValueError(
                    f"Limit and threshold must be 1D variables with "
                    f"dimension equal to `vector.dof()` ({vector.dof()})."
                )
        HingeCost._check_limits(self.down_limit, self.up_limit)
        if self.threshold.tensor.max() < 0.0:
            raise ValueError("Threshold values must be positive numbers.")
        self.register_optim_var("vector")
        self.register_aux_vars(["down_limit", "up_limit", "threshold"])

    @staticmethod
    def _check_limits(down_limit: Variable, up_limit: Variable):
        dof = down_limit.shape[1]
        for i in range(dof):
            if not (down_limit[:, i] <= up_limit[:, i]).all():
                raise ValueError("All down_limit must be <= than up_limit.")

    @staticmethod
    def _convert_to_tensor_if_float(
        value: Union[float, torch.Tensor, Variable], dof: int
    ) -> Union[torch.Tensor, Variable]:
        if isinstance(value, float):
            return torch.ones(1, dof) * value
        return value

    def dim(self):
        return self.vector.dof()

    def _compute_error(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vector = self.vector.tensor
        threshold = self.threshold.tensor
        down_limit = self.down_limit.tensor + threshold
        up_limit = self.up_limit.tensor - threshold
        below_idx = vector < down_limit
        above_idx = vector > up_limit
        error = vector.new_zeros(vector.shape)
        error = torch.where(below_idx, down_limit - vector, error)
        error = torch.where(above_idx, vector - up_limit, error)
        return error, below_idx, above_idx

    def error(self) -> torch.Tensor:
        return self._compute_error()[0]

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        error, below_idx, above_idx = self._compute_error()
        J = self.vector.tensor.new_zeros(
            self.vector.shape[0], self.vector.dof(), self.vector.dof()
        )
        J[below_idx.diag_embed()] = -1.0
        J[above_idx.diag_embed()] = 1.0
        return [J], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "HingeCost":
        return HingeCost(
            self.vector.copy(),
            self.down_limit.copy(),
            self.up_limit.copy(),
            self.threshold.copy(),
            self.weight.copy(),
            name=new_name,
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
                "Nonholonomic only accepts 3D velocity or poses (x, y, theta dims). "
                "Poses can either be SE2 or Vector variables. Velocities only Vector."
            )
        self.pose = pose
        self.vel = vel
        self.register_optim_vars(["pose", "vel"])
        self.weight = cost_weight
        is_se2 = isinstance(self.pose, SE2)
        self._compute_error_impl = (
            self._compute_error_se2_impl if is_se2 else self._compute_error_vector_impl
        )
        self._compute_jacobians = (
            self._compute_jacobians_se2 if is_se2 else self._compute_jacobians_vector
        )

    def dim(self):
        return 1

    def _compute_error_se2_impl(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.vel[:, 1], None, None

    def _compute_error_vector_impl(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        cos = self.pose[:, 2].cos()
        sin = self.pose[:, 2].sin()
        error = self.vel[:, 1] * cos - self.vel[:, 0] * sin
        return error, cos, sin

    def _compute_error(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        error, cos, sin = self._compute_error_impl()
        return error.view(-1, 1), cos, sin

    def error(self) -> torch.Tensor:
        return self._compute_error()[0]

    def _compute_jacobians_se2(
        self,
        error: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Jpose = error.new_zeros(error.shape[0], 1, 3)
        Jvel = error.new_zeros(error.shape[0], 1, 3)
        Jvel[:, 0, 1] = 1
        return Jpose, Jvel

    def _compute_jacobians_vector(
        self,
        error: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Jpose = self.vel.tensor.new_zeros(self.vel.shape[0], 1, 3)
        Jvel = self.vel.tensor.new_zeros(self.vel.shape[0], 1, 3)
        Jpose[:, 0, 2] = -(self.vel[:, 1] * sin + self.vel[:, 0] * cos)
        Jvel[:, 0, 0] = -sin
        Jvel[:, 0, 1] = cos
        return Jpose, Jvel

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Pre-allocate jacobian tensors
        batch_size = self.vel.shape[0]
        Jvel = self.vel.tensor.new_zeros(batch_size, 1, 3)
        error, cos, sin = self._compute_error()
        Jpose, Jvel = self._compute_jacobians(error, cos, sin)
        return [Jpose, Jvel], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "Nonholonomic":
        return Nonholonomic(
            self.pose.copy(),
            self.vel.copy(),
            self.weight.copy(),
            name=new_name,
        )
