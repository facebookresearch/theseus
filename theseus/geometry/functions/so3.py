# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from theseus.geometry.lie_group_check import _LieGroupCheckContext

from typing import List, Optional, cast

import torch

import theseus.constants


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


def check_hat_matrix(matrix: torch.Tensor):
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError("Hat matrices of SO(3) can only be 3x3 matrices")

    checks_enabled, silent_unchecks = _LieGroupCheckContext.get_context()
    if checks_enabled:
        if (
            matrix.transpose(1, 2) + matrix
        ).abs().max().item() > theseus.constants._SO3_HAT_EPS[matrix.dtype]:
            raise ValueError("Hat matrices of SO(3) can only be skew-symmetric.")
    elif not silent_unchecks:
        warnings.warn(
            "Lie group checks are disabled, so the skew-symmetry of hat matrices is "
            "not checked for SO3.",
            RuntimeWarning,
        )


class hat(torch.autograd.Function):
    @staticmethod
    def call(tangent_vector: torch.Tensor) -> torch.Tensor:
        if not check_tangent_vector(tangent_vector):
            raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
        matrix = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 3)

        matrix[:, 0, 1] = -tangent_vector[:, 2].view(-1)
        matrix[:, 0, 2] = tangent_vector[:, 1].view(-1)
        matrix[:, 1, 2] = -tangent_vector[:, 0].view(-1)
        matrix[:, 1, 0] = tangent_vector[:, 2].view(-1)
        matrix[:, 2, 0] = -tangent_vector[:, 1].view(-1)
        matrix[:, 2, 1] = tangent_vector[:, 0].view(-1)

        return matrix

    @staticmethod
    def forward(ctx, tangent_vector):
        return hat.call(tangent_vector)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return torch.stack(
            (
                grad_output[:, 2, 1] - grad_output[:, 1, 2],
                grad_output[:, 0, 2] - grad_output[:, 2, 0],
                grad_output[:, 1, 0] - grad_output[:, 0, 1],
            ),
            dim=1,
        )


class exp_map(torch.autograd.Function):
    @staticmethod
    def call(
        tangent_vector: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ):
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
            if len(jacobians) != 0:
                raise ValueError("jacobians list to be populated must be empty.")
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

    @staticmethod
    def jacobian(tangent_vector: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def forward(
        ctx,
        tangent_vector,
        jacobians=None,
    ):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = exp_map.call(tangent_vector, jacobians)
        ctx.save_for_backward(tangent_vector)

        if jacobians is not None:
            ctx.save_for_backward(jacobians[0])

        return ret

    @staticmethod
    def backward(ctx, grad_output):
        saved_tensors = ctx.saved_tensors

        if len(saved_tensors) == 1:
            tangent_vector: torch.Tensor = saved_tensors[0]
            ctx.save_for_backward(exp_map(tangent_vector))

        jacs = ctx.saved_tensors[1]

        return jacs
