# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import torch

from theseus.core import CostFunction, CostWeight, Variable, as_variable
from theseus.geometry import LieGroup, Vector


class DoubleIntegrator(CostFunction):
    def __init__(
        self,
        pose1: LieGroup,
        vel1: Vector,
        pose2: LieGroup,
        vel2: Vector,
        dt: Union[float, torch.Tensor, Variable],
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        dof = pose1.dof()
        if not (vel1.dof() == pose2.dof() == vel2.dof() == dof):
            raise ValueError(
                "All variables for a DoubleIntegrator must have the same dimension."
            )
        self.dt = as_variable(dt)
        if self.dt.tensor.squeeze().ndim > 1:
            raise ValueError(
                "dt data must be a 0-D or 1-D tensor with numel in {1, batch_size}."
            )
        self.dt.tensor = self.dt.tensor.view(-1, 1)
        self.pose1 = pose1
        self.vel1 = vel1
        self.pose2 = pose2
        self.vel2 = vel2
        self.register_optim_vars(["pose1", "vel1", "pose2", "vel2"])
        self.register_aux_vars(["dt"])
        self.weight = cost_weight

    def dim(self):
        return 2 * self.pose1.dof()

    def _new_pose_diff(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return self.pose1.local(self.pose2, jacobians=jacobians)

    def _error_from_pose_diff(self, pose_diff: torch.Tensor) -> torch.Tensor:
        pose_diff_err = pose_diff - self.dt.tensor.view(-1, 1) * self.vel1.tensor
        vel_diff = self.vel2.tensor - self.vel1.tensor
        return torch.cat([pose_diff_err, vel_diff], dim=1)

    def error(self) -> torch.Tensor:
        return self._error_from_pose_diff(self._new_pose_diff())

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Pre-allocate jacobian tensors
        batch_size = self.pose1.shape[0]
        dof = self.pose1.dof()
        dtype = self.pose1.dtype
        device = self.pose1.device
        Jerr_pose1 = torch.zeros(batch_size, 2 * dof, dof, dtype=dtype, device=device)
        Jerr_vel1 = torch.zeros_like(Jerr_pose1)
        Jerr_pose2 = torch.zeros_like(Jerr_pose1)
        Jerr_vel2 = torch.zeros_like(Jerr_pose1)

        Jlocal: List[torch.Tensor] = []
        error = self._error_from_pose_diff(self._new_pose_diff(Jlocal))
        Jerr_pose1[:, :dof, :] = Jlocal[0]
        identity = torch.eye(dof, dtype=dtype, device=device).repeat(batch_size, 1, 1)
        Jerr_vel1[:, :dof, :] = -self.dt.tensor.view(-1, 1, 1) * identity
        Jerr_vel1[:, dof:, :] = -identity
        Jerr_pose2[:, :dof, :] = Jlocal[1]
        Jerr_vel2[:, dof:, :] = identity
        return [Jerr_pose1, Jerr_vel1, Jerr_pose2, Jerr_vel2], error

    def _copy_impl(self, new_name: Optional[str] = None) -> "DoubleIntegrator":
        return DoubleIntegrator(
            self.pose1.copy(),
            self.vel1.copy(),
            self.pose2.copy(),
            self.vel2.copy(),
            self.dt.copy(),
            self.weight.copy(),
            name=new_name,
        )


class GPCostWeight(CostWeight):
    # Qc_inv is either a single square matrix or a batch of square matrices
    # dt is either a single element or a 1-D batch
    def __init__(
        self,
        Qc_inv: Union[Variable, torch.Tensor],
        dt: Union[float, Variable, torch.Tensor],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        dt = as_variable(dt)
        if dt.tensor.squeeze().ndim > 1:
            raise ValueError("dt must be a 0-D or 1-D tensor.")
        self.dt = dt
        self.dt.tensor = self.dt.tensor.view(-1, 1)
        if not (self.dt.tensor > 0).all():
            raise ValueError("dt must be greater than 0.")

        Qc_inv = as_variable(Qc_inv)
        if Qc_inv.ndim not in [2, 3]:
            raise ValueError("Qc_inv must be a single matrix or a batch of matrices.")
        if not Qc_inv.shape[-2] == Qc_inv.shape[-1]:
            raise ValueError("Qc_inv must contain square matrices.")
        self.Qc_inv = Qc_inv
        self.Qc_inv.tensor = (
            Qc_inv.tensor if Qc_inv.ndim == 3 else Qc_inv.tensor.unsqueeze(0)
        )
        try:
            torch.linalg.cholesky(Qc_inv.tensor)
        except RuntimeError:
            raise ValueError("Qc_inv must be positive definite.")

        self.register_aux_vars(["Qc_inv", "dt"])

    def is_zero(self) -> torch.Tensor:
        return torch.zeros(self.Qc_inv.shape[0]).bool()

    def _compute_cost_weight(self) -> torch.Tensor:
        batch_size, dof, _ = self.Qc_inv.shape
        cost_weight = torch.empty(
            batch_size,
            2 * dof,
            2 * dof,
            dtype=self.Qc_inv.dtype,
            device=self.Qc_inv.device,
        )
        dt_data = self.dt.tensor.view(-1, 1, 1)
        Q11 = 12.0 * dt_data.pow(-3.0) * self.Qc_inv.tensor
        Q12 = -6.0 * dt_data.pow(-2.0) * self.Qc_inv.tensor
        Q22 = 4.0 * dt_data.reciprocal() * self.Qc_inv.tensor
        cost_weight[:, :dof, :dof] = Q11
        cost_weight[:, :dof:, dof:] = Q12
        cost_weight[:, dof:, :dof] = Q12
        cost_weight[:, dof:, dof:] = Q22
        return (
            torch.linalg.cholesky(cost_weight.transpose(-2, -1).conj())
            .transpose(-2, -1)
            .conj()
        )

    def weight_error(self, error: torch.Tensor) -> torch.Tensor:
        weights = self._compute_cost_weight()
        return torch.matmul(weights, error.unsqueeze(2)).squeeze(2)

    def weight_jacobians_and_error(
        self,
        jacobians: List[torch.Tensor],
        error: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cost_weight = self._compute_cost_weight()
        error = torch.matmul(cost_weight, error.unsqueeze(2)).squeeze(2)
        new_jacobians = []
        for jac in jacobians:
            # Jacobian is batch_size x cost_function_dim x var_dim
            # This left multiplies the weights (inv cov.) to jacobian
            new_jacobians.append(torch.matmul(cost_weight, jac))
        return new_jacobians, error

    def _copy_impl(self, new_name: Optional[str] = None) -> "GPCostWeight":
        # need to pass data, since it could be Parameter(self.Qc_inv) in which
        # case Qc_inv won't be up to date.
        # This will change with the "learning happens outside API"
        return GPCostWeight(self.Qc_inv.copy(), self.dt.copy(), name=new_name)


class GPMotionModel(DoubleIntegrator):
    def __init__(
        self,
        pose1: LieGroup,
        vel1: Vector,
        pose2: LieGroup,
        vel2: Vector,
        dt: Union[float, Variable, torch.Tensor],
        cost_weight: GPCostWeight,
        name: Optional[str] = None,
    ):
        if not isinstance(cost_weight, GPCostWeight):
            raise ValueError(
                "GPMotionModel only accepts cost weights of type GPCostWeight. "
                "For other weight types, consider using DoubleIntegrator instead."
            )
        self.dt = as_variable(dt)
        if self.dt.tensor.squeeze().ndim > 1:
            raise ValueError("dt must be a 0-D or 1-D tensor.")
        self.dt.tensor = self.dt.tensor.view(-1, 1)
        super().__init__(pose1, vel1, pose2, vel2, dt, cost_weight, name=name)

    def _copy_impl(self, new_name: Optional[str] = None) -> "GPMotionModel":
        return cast(GPMotionModel, super()._copy_impl(new_name=new_name))
