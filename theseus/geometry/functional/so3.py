# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import cast, Tuple

from . import constants
from . import lie_group as LieGroup
from .utils import get_module


NAME: str = "SO3"
DIM: int = 3


def check_group_tensor(tensor: torch.Tensor) -> bool:
    with torch.no_grad():
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
            raise ValueError("SO3 data tensors can only be 3x3 matrices.")

        MATRIX_EPS = constants._SO3_MATRIX_EPS[tensor.dtype]
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


def _exp_map_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    if not check_tangent_vector(tangent_vector):
        raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
    tangent_vector = tangent_vector.view(-1, 3)
    theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
    theta2 = theta**2
    # Compute the approximations when theta ~ 0
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)

    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine = theta.sin()
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
    )
    ret = (
        one_minus_cosine_by_theta2
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

    return ret


def _jexp_map_impl(tangent_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not check_tangent_vector(tangent_vector):
        raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
    tangent_vector = tangent_vector.view(-1, 3)
    theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
    theta2 = theta**2
    # Compute the approximations when theta ~ 0
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine = theta.sin()
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
    )
    theta3_nz = theta_nz * theta2_nz
    theta_minus_sine_by_theta3 = torch.where(
        near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
    )

    ret = (
        one_minus_cosine_by_theta2
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

    jac = (
        theta_minus_sine_by_theta3
        * tangent_vector.view(-1, 3, 1)
        @ tangent_vector.view(-1, 1, 3)
    )
    diag_jac = jac.diagonal(dim1=1, dim2=2)
    diag_jac += sine_by_theta.view(-1, 1)
    jac_temp = one_minus_cosine_by_theta2.view(-1, 1) * tangent_vector
    jac[:, 0, 1] += jac_temp[:, 2]
    jac[:, 1, 0] -= jac_temp[:, 2]
    jac[:, 0, 2] -= jac_temp[:, 1]
    jac[:, 2, 0] += jac_temp[:, 1]
    jac[:, 1, 2] += jac_temp[:, 0]
    jac[:, 2, 1] -= jac_temp[:, 0]

    return jac, ret


class ExpMap(LieGroup.UnaryOperator):
    @classmethod
    def forward(cls, ctx, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _exp_map_impl(tangent_vector)
        ctx.save_for_backward(tangent_vector, ret)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        tangent_vector: torch.Tensor = ctx.saved_tensors[0]
        group: torch.Tensor = ctx.saved_tensors[1]
        if not hasattr(ctx, "jacobians"):
            ctx.jacobians: torch.Tensor = _jexp_map_impl(tangent_vector)[0]
        jacs = ctx.jacobians
        dR = group.transpose(1, 2) @ grad_output
        grad_input = jacs.transpose(1, 2) @ torch.stack(
            (
                dR[:, 2, 1] - dR[:, 1, 2],
                dR[:, 0, 2] - dR[:, 2, 0],
                dR[:, 1, 0] - dR[:, 0, 1],
            ),
            dim=1,
        ).view(-1, 3, 1)
        return grad_input.view(-1, 3)


_module = get_module(__name__)

_exp_map_autograd_fn = ExpMap.apply
_jexp_map_autograd_fn = _jexp_map_impl

exp_map, jexp_map = LieGroup.UnaryOperatorFactory(_module, "exp_map")
