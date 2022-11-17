# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import theseus

from typing import Optional, List, cast

from .lie_group_function import LieGroupExpMap
from .utils import check_jacobians_list


def name() -> str:
    return "SO3"


def dim() -> int:
    return 3


def check_group_tensor(tensor: torch.Tensor) -> bool:
    with torch.no_grad():
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
            raise ValueError("SO3 data tensors can only be 3x3 matrices.")

        MATRIX_EPS = theseus.constants._SO3_MATRIX_EPS[tensor.dtype]
        if tensor.dtype != torch.float64:
            tensor = tensor.double()

        _check = (
            torch.matmul(tensor, tensor.transpose(1, 2))
            - torch.eye(3, 3, dtype=tensor.dtype, device=tensor.device)
        ).abs().max().item() < MATRIX_EPS
        _check &= (torch.linalg.det(tensor) - 1).abs().max().item() < MATRIX_EPS

    return _check


def check_tangent_vector(tangent_vector: torch.Tensor) -> bool:
    _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (3, 1)
    _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 3
    return _check


class ExpMap(LieGroupExpMap):
    @classmethod
    def call(
        cls,
        tangent_vector: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not check_tangent_vector(tangent_vector):
            raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
        tangent_vector = tangent_vector.view(-1, 3)
        ret = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 3)
        theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
        theta2 = theta**2
        # Compute the approximations when theta ~ 0
        near_zero = theta < theseus.constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)

        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine = theta.sin()
        sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
        one_minus_cosie_by_theta2 = torch.where(
            near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
        )
        ret = (
            one_minus_cosie_by_theta2
            * tangent_vector.view(-1, 3, 1)
            @ tangent_vector.view(-1, 1, 3)
        )

        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        sine_axis = sine_by_theta.view(-1, 1) * tangent_vector
        ret[:, 0, 1] -= sine_axis[:, 2]
        ret[:, 1, 0] += sine_axis[:, 2]
        ret[:, 0, 2] += sine_axis[:, 1]
        ret[:, 2, 0] -= sine_axis[:, 1]
        ret[:, 1, 2] -= sine_axis[:, 0]
        ret[:, 2, 1] += sine_axis[:, 0]

        if jacobians is not None:
            check_jacobians_list(jacobians)
            theta3_nz = theta_nz * theta2_nz
            theta_minus_sine_by_theta3 = torch.where(
                near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
            )
            jac = (
                theta_minus_sine_by_theta3
                * tangent_vector.view(-1, 3, 1)
                @ tangent_vector.view(-1, 1, 3)
            )
            diag_jac = jac.diagonal(dim1=1, dim2=2)
            diag_jac += sine_by_theta.view(-1, 1)

            jac_temp = one_minus_cosie_by_theta2.view(-1, 1) * tangent_vector

            jac[:, 0, 1] += jac_temp[:, 2]
            jac[:, 1, 0] -= jac_temp[:, 2]
            jac[:, 0, 2] -= jac_temp[:, 1]
            jac[:, 2, 0] += jac_temp[:, 1]
            jac[:, 1, 2] += jac_temp[:, 0]
            jac[:, 2, 1] -= jac_temp[:, 0]

            jacobians.append(jac)

        return ret

    @classmethod
    def jacobian(cls, tangent_vector: torch.Tensor) -> torch.Tensor:
        if not check_tangent_vector(tangent_vector):
            raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
        tangent_vector = tangent_vector.view(-1, 3)
        theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
        theta2 = theta**2
        # Compute the approximations when theta ~ 0
        near_zero = theta < theseus.constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)
        theta3_nz = theta_nz * theta2_nz
        sine = theta.sin()
        theta_minus_sine_by_theta3 = torch.where(
            near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
        )
        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine = theta.sin()
        sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
        one_minus_cosie_by_theta2 = torch.where(
            near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
        )

        jac = (
            theta_minus_sine_by_theta3
            * tangent_vector.view(-1, 3, 1)
            @ tangent_vector.view(-1, 1, 3)
        )
        diag_jac = jac.diagonal(dim1=1, dim2=2)
        diag_jac += sine_by_theta.view(-1, 1)

        jac_temp = one_minus_cosie_by_theta2.view(-1, 1) * tangent_vector

        jac[:, 0, 1] += jac_temp[:, 2]
        jac[:, 1, 0] -= jac_temp[:, 2]
        jac[:, 0, 2] -= jac_temp[:, 1]
        jac[:, 2, 0] += jac_temp[:, 1]
        jac[:, 1, 2] += jac_temp[:, 0]
        jac[:, 2, 1] -= jac_temp[:, 0]

        return jac

    @classmethod
    def forward(
        cls,
        ctx,
        tangent_vector,
        jacobians=None,
    ):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = cls.call(tangent_vector, jacobians)
        ctx.save_for_backward(tangent_vector, ret)
        ctx.jacobians = jacobians

        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        if ctx.jacobians is None:
            tangent_vector: torch.Tensor = ctx.saved_tensors[0]
            ctx.jacobians = cls.jacobian(tangent_vector)

        R: torch.Tensor = ctx.saved_tensors[1]
        jacs: torch.Tensor = ctx.jacobians
        dR = R.transpose(1, 2) @ grad_output
        grad = jacs.transpose(1, 2) @ torch.stack(
            (
                dR[:, 2, 1] - dR[:, 1, 2],
                dR[:, 0, 2] - dR[:, 2, 0],
                dR[:, 1, 0] - dR[:, 0, 1],
            ),
            dim=1,
        ).view(-1, 3, 1)
        return grad.view(-1, 3)


exp_map = ExpMap.apply
