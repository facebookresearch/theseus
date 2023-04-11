# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import cast, List, Tuple, Optional

from . import constants
from . import lie_group, so3_impl as SO3
from .utils import get_module


NAME: str = "SE3"
DIM: int = 6


_module = get_module(__name__)


def check_group_tensor(tensor: torch.Tensor):
    with torch.no_grad():
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 4):
            raise ValueError(
                f"SE3 data tensors can only be 3x4 matrices, but got shape {tensor.shape}."
            )
    SO3.check_group_tensor(tensor[:, :, :3])


def check_transform_tensor(tensor: torch.Tensor):
    SO3.check_transform_tensor(tensor)


def check_tangent_vector(tangent_vector: torch.Tensor):
    _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (6, 1)
    _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 6
    if not _check:
        raise ValueError(
            f"Tangent vectors of SE3 should be 6-D vectors, but got shape {tangent_vector.shape}."
        )


def check_hat_matrix(matrix: torch.Tensor):
    if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
        raise ValueError("Hat matrices of SE(3) can only be 3x4 matrices")

    if matrix[:, -1].abs().max() > constants._SE3_NEAR_ZERO_EPS[matrix.dtype]:
        raise ValueError("The last row for hat matrices of SE(3) must be zero")

    SO3.check_hat_matrix(matrix[:, :3, :3])


def check_lift_matrix(matrix: torch.Tensor):
    return matrix.shape[-1] == 6


def check_project_matrix(matrix: torch.Tensor):
    return matrix.shape[-2:] == (3, 4)


def check_left_act_matrix(matrix: torch.Tensor):
    if matrix.shape[-2] != 3:
        raise ValueError("Inconsistent shape for the matrix.")


def check_left_project_matrix(matrix: torch.Tensor):
    if matrix.shape[-2:] != (3, 4):
        raise ValueError("Inconsistent shape for the matrix.")


# -----------------------------------------------------------------------------
# Rand
# -----------------------------------------------------------------------------
# TODO: Remove duplicate code between rand and randn
def rand(
    *size: int,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: constants.DeviceType = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    if len(size) != 1:
        raise ValueError("The size should be 1D.")
    rotation = SO3.rand(size[0], generator=generator, dtype=dtype, device=device)
    translation = torch.rand(
        size[0], 3, 1, generator=generator, dtype=dtype, device=device
    )
    ret = torch.cat((rotation, translation), dim=2)
    ret.requires_grad_(requires_grad)
    return ret


# -----------------------------------------------------------------------------
# Randn
# -----------------------------------------------------------------------------
def randn(
    *size: int,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: constants.DeviceType = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    if len(size) != 1:
        raise ValueError("The size should be 1D.")
    rotation = SO3.randn(size[0], generator=generator, dtype=dtype, device=device)
    translation = torch.randn(
        size[0], 3, 1, generator=generator, dtype=dtype, device=device
    )
    ret = torch.cat((rotation, translation), dim=2)
    ret.requires_grad_(requires_grad)
    return ret


# -----------------------------------------------------------------------------
# Identity
# -----------------------------------------------------------------------------
_BASE_IDENTITY_SE3 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]


def identity(
    *size: int,
    dtype: Optional[torch.dtype] = None,
    device: constants.DeviceType = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    if len(size) != 1:
        raise ValueError("The size should be 1D.")
    ret = torch.tensor(_BASE_IDENTITY_SE3, dtype=dtype, device=device).repeat(
        size[0], 1, 1
    )
    ret.requires_grad_(requires_grad)
    return ret


# -----------------------------------------------------------------------------
# Exponential Map
# -----------------------------------------------------------------------------
def _exp_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    check_tangent_vector(tangent_vector)
    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < constants._SE3_NEAR_ZERO_EPS[tangent_vector.dtype]
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, theta.sin() / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
    )
    ret = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 4)
    ret[:, :3, :3] = (
        one_minus_cosine_by_theta2
        * tangent_vector_ang
        @ tangent_vector_ang.transpose(1, 2)
    )

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2
    )
    theta_minus_sine_by_theta3_t = torch.where(
        near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
    )

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(
        tangent_vector_ang, tangent_vector_lin, dim=1
    )
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )

    return ret


def _jexp_impl(
    tangent_vector: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_tangent_vector(tangent_vector)
    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < constants._SE3_NEAR_ZERO_EPS[tangent_vector.dtype]
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, theta.sin() / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
    )
    ret = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 4)
    ret[:, :3, :3] = (
        one_minus_cosine_by_theta2
        * tangent_vector_ang
        @ tangent_vector_ang.transpose(1, 2)
    )

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2
    )
    theta_minus_sine_by_theta3_t = torch.where(
        near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
    )

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(
        tangent_vector_ang, tangent_vector_lin, dim=1
    )
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )

    # compute jacobians
    theta3_nz = theta_nz * theta2_nz
    theta_minus_sine_by_theta3_rot = torch.where(
        near_zero, torch.zeros_like(theta), theta_minus_sine_by_theta3_t
    )
    jac = tangent_vector.new_zeros(
        tangent_vector.shape[0],
        6,
        6,
    )
    jac[:, :3, :3] = (
        theta_minus_sine_by_theta3_rot
        * tangent_vector_ang.view(-1, 3, 1)
        @ tangent_vector_ang.view(-1, 1, 3)
    )
    diag_jac = jac.diagonal(dim1=1, dim2=2)
    diag_jac += sine_by_theta.view(-1, 1)

    jac_temp_rot = one_minus_cosine_by_theta2.view(-1, 1) * tangent_vector_ang.view(
        -1, 3
    )

    jac[:, 0, 1] += jac_temp_rot[:, 2]
    jac[:, 1, 0] -= jac_temp_rot[:, 2]
    jac[:, 0, 2] -= jac_temp_rot[:, 1]
    jac[:, 2, 0] += jac_temp_rot[:, 1]
    jac[:, 1, 2] += jac_temp_rot[:, 0]
    jac[:, 2, 1] -= jac_temp_rot[:, 0]

    jac[:, 3:, 3:] = jac[:, :3, :3]

    minus_one_by_twelve = torch.tensor(
        -1 / 12.0,
        dtype=sine_by_theta.dtype,
        device=sine_by_theta.device,
    )
    d_one_minus_cosine_by_theta2 = torch.where(
        near_zero,
        minus_one_by_twelve,
        (sine_by_theta - 2 * one_minus_cosine_by_theta2) / theta2_nz,
    )
    minus_one_by_sixty = torch.tensor(
        -1 / 60.0,
        dtype=one_minus_cosine_by_theta2.dtype,
        device=one_minus_cosine_by_theta2.device,
    )
    d_theta_minus_sine_by_theta3 = torch.where(
        near_zero,
        minus_one_by_sixty,
        (one_minus_cosine_by_theta2 - 3 * theta_minus_sine_by_theta3_t) / theta2_nz,
    )

    w = tangent_vector[:, 3:]
    v = tangent_vector[:, :3]
    wv = w.cross(v, dim=1)
    wwv = w.cross(wv, dim=1)
    sw = theta_minus_sine_by_theta3_t.view(-1, 1) * w

    jac_temp_t = (
        d_one_minus_cosine_by_theta2.view(-1, 1) * wv
        + d_theta_minus_sine_by_theta3.view(-1, 1) * wwv
    ).view(-1, 3, 1) @ w.view(-1, 1, 3)
    jac_temp_t -= v.view(-1, 3, 1) @ sw.view(-1, 1, 3)
    jac_temp_v = (
        -one_minus_cosine_by_theta2.view(-1, 1) * v
        - theta_minus_sine_by_theta3_t.view(-1, 1) * wv
    )
    jac_temp_t += SO3._hat_autograd_fn(jac_temp_v)
    diag_jac_t = torch.diagonal(jac_temp_t, dim1=1, dim2=2)
    diag_jac_t += (sw.view(-1, 1, 3) @ v.view(-1, 3, 1)).view(-1, 1)

    jac[:, :3, 3:] = ret[:, :, :3].transpose(1, 2) @ jac_temp_t

    return [jac], ret


class Exp(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _exp_impl(tangent_vector)
        ctx.save_for_backward(tangent_vector, ret)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        tangent_vector: torch.Tensor = ctx.saved_tensors[0]
        group: torch.Tensor = ctx.saved_tensors[1]
        if not hasattr(ctx, "jacobians"):
            ctx.jacobians: torch.Tensor = _jexp_impl(tangent_vector)[0][0]
        jacs = ctx.jacobians
        dg = group[:, :, :3].transpose(1, 2) @ grad_output
        grad_input = jacs.transpose(1, 2) @ torch.stack(
            (
                dg[:, 0, 3],
                dg[:, 1, 3],
                dg[:, 2, 3],
                dg[:, 2, 1] - dg[:, 1, 2],
                dg[:, 0, 2] - dg[:, 2, 0],
                dg[:, 1, 0] - dg[:, 0, 1],
            ),
            dim=1,
        ).view(-1, 6, 1)
        return grad_input.view(-1, 6)


# TODO: Implement analytic backward for _jexp_impl
_exp_autograd_fn = Exp.apply
_jexp_autograd_fn = _jexp_impl


# -----------------------------------------------------------------------------
# Logarithm Map
# -----------------------------------------------------------------------------
def _log_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    sine_axis = group.new_zeros(group.shape[0], 3)
    sine_axis[:, 0] = 0.5 * (group[:, 2, 1] - group[:, 1, 2])
    sine_axis[:, 1] = 0.5 * (group[:, 0, 2] - group[:, 2, 0])
    sine_axis[:, 2] = 0.5 * (group[:, 1, 0] - group[:, 0, 1])
    cosine = 0.5 * (group[:, 0, 0] + group[:, 1, 1] + group[:, 2, 2] - 1)
    sine = sine_axis.norm(dim=1)
    theta = torch.atan2(sine, cosine)
    theta2 = theta**2
    non_zero = torch.ones(1, dtype=group.dtype, device=group.device)

    near_zero = theta < constants._SE3_NEAR_ZERO_EPS[group.dtype]
    near_pi = 1 + cosine <= constants._SE3_NEAR_PI_EPS[group.dtype]

    # Compute the rotation
    near_zero_or_near_pi = torch.logical_or(near_zero, near_pi)
    # Compute the approximation of theta / sin(theta) when theta is near to 0
    sine_nz = torch.where(near_zero_or_near_pi, non_zero, sine)
    scale = torch.where(
        near_zero_or_near_pi,
        1 + sine**2 / 6,
        theta / sine_nz,
    )
    ret_ang = sine_axis * scale.view(-1, 1)

    # theta is near pi
    ddiag = torch.diagonal(group, dim1=1, dim2=2)
    # Find the index of major coloumns and diagonals
    major = torch.logical_and(
        ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
    ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
    aux = torch.ones(group.shape[0], dtype=torch.bool)
    sel_rows = 0.5 * (group[aux, major, :3] + group[aux, :3, major])
    sel_rows[aux, major] -= cosine
    axis = sel_rows / torch.where(
        near_zero.view(-1, 1),
        non_zero.view(-1, 1),
        sel_rows.norm(dim=1, keepdim=True),
    )
    sign_tmp = sine_axis[aux, major].sign()
    sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
    ret_ang = torch.where(
        near_pi.view(-1, 1), axis * (theta * sign).view(-1, 1), ret_ang
    )

    # Compute the translation
    sine_theta = sine * theta
    two_cosine_minus_two = 2 * cosine - 2
    two_cosine_minus_two_nz = torch.where(near_zero, non_zero, two_cosine_minus_two)

    theta2_nz = torch.where(near_zero, non_zero, theta2)

    a = torch.where(near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz)
    b = torch.where(
        near_zero,
        1.0 / 12 + theta2 / 720,
        (sine_theta + two_cosine_minus_two) / (theta2_nz * two_cosine_minus_two_nz),
    )

    translation = group[:, :, 3].view(-1, 3, 1)
    ret_lin = a.view(-1, 1) * group[:, :, 3]
    ret_lin -= 0.5 * torch.cross(ret_ang, group[:, :, 3], dim=1)
    ret_ang_ext = ret_ang.view(-1, 3, 1)
    ret_lin += b.view(-1, 1) * (
        ret_ang_ext @ (ret_ang_ext.transpose(1, 2) @ translation)
    ).view(-1, 3)

    return torch.cat([ret_lin, ret_ang], dim=1)


def _jlog_impl(group: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group)
    sine_axis = group.new_zeros(group.shape[0], 3)
    sine_axis[:, 0] = 0.5 * (group[:, 2, 1] - group[:, 1, 2])
    sine_axis[:, 1] = 0.5 * (group[:, 0, 2] - group[:, 2, 0])
    sine_axis[:, 2] = 0.5 * (group[:, 1, 0] - group[:, 0, 1])
    cosine = 0.5 * (group[:, 0, 0] + group[:, 1, 1] + group[:, 2, 2] - 1)
    sine = sine_axis.norm(dim=1)
    theta = torch.atan2(sine, cosine)
    theta2 = theta**2
    non_zero = torch.ones(1, dtype=group.dtype, device=group.device)

    near_zero = theta < constants._SE3_NEAR_ZERO_EPS[group.dtype]
    near_pi = 1 + cosine <= constants._SE3_NEAR_PI_EPS[group.dtype]

    # Compute the rotation
    near_zero_or_near_pi = torch.logical_or(near_zero, near_pi)
    # Compute the approximation of theta / sin(theta) when theta is near to 0
    sine_nz = torch.where(near_zero_or_near_pi, non_zero, sine)
    scale = torch.where(
        near_zero_or_near_pi,
        1 + sine**2 / 6,
        theta / sine_nz,
    )
    ret_ang = sine_axis * scale.view(-1, 1)

    # theta is near pi
    ddiag = torch.diagonal(group, dim1=1, dim2=2)
    # Find the index of major coloumns and diagonals
    major = torch.logical_and(
        ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
    ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
    aux = torch.ones(group.shape[0], dtype=torch.bool)
    sel_rows = 0.5 * (group[aux, major, :3] + group[aux, :3, major])
    sel_rows[aux, major] -= cosine
    axis = sel_rows / torch.where(
        near_zero.view(-1, 1),
        non_zero.view(-1, 1),
        sel_rows.norm(dim=1, keepdim=True),
    )
    sign_tmp = sine_axis[aux, major].sign()
    sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
    ret_ang = torch.where(
        near_pi.view(-1, 1), axis * (theta * sign).view(-1, 1), ret_ang
    )

    # Compute the translation
    sine_theta = sine * theta
    two_cosine_minus_two = 2 * cosine - 2
    two_cosine_minus_two_nz = torch.where(near_zero, non_zero, two_cosine_minus_two)

    theta2_nz = torch.where(near_zero, non_zero, theta2)

    a = torch.where(near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz)
    b = torch.where(
        near_zero,
        1.0 / 12 + theta2 / 720,
        (sine_theta + two_cosine_minus_two) / (theta2_nz * two_cosine_minus_two_nz),
    )

    translation = group[:, :, 3].view(-1, 3, 1)
    ret_lin = a.view(-1, 1) * group[:, :, 3]
    ret_lin -= 0.5 * torch.cross(ret_ang, group[:, :, 3], dim=1)
    ret_ang_ext = ret_ang.view(-1, 3, 1)
    ret_lin += b.view(-1, 1) * (
        ret_ang_ext @ (ret_ang_ext.transpose(1, 2) @ translation)
    ).view(-1, 3)

    # jacobians
    jac = group.new_zeros(group.shape[0], 6, 6)

    b_ret_ang = b.view(-1, 1) * ret_ang
    jac[:, :3, :3] = b_ret_ang.view(-1, 3, 1) * ret_ang.view(-1, 1, 3)

    half_ret_ang = 0.5 * ret_ang
    jac[:, 0, 1] -= half_ret_ang[:, 2]
    jac[:, 1, 0] += half_ret_ang[:, 2]
    jac[:, 0, 2] += half_ret_ang[:, 1]
    jac[:, 2, 0] -= half_ret_ang[:, 1]
    jac[:, 1, 2] -= half_ret_ang[:, 0]
    jac[:, 2, 1] += half_ret_ang[:, 0]

    diag_jac_rot = torch.diagonal(jac[:, :3, :3], dim1=1, dim2=2)
    diag_jac_rot += a.view(-1, 1)

    jac[:, 3:, 3:] = jac[:, :3, :3]

    theta_nz = torch.where(near_zero, non_zero, theta)
    theta4_nz = theta2_nz**2
    c = torch.where(
        near_zero,
        -1 / 360.0 - theta2 / 7560.0,
        -(2 * two_cosine_minus_two + theta * sine + theta2)
        / (theta4_nz * two_cosine_minus_two_nz),
    )
    d = torch.where(
        near_zero,
        -1 / 6.0 - theta2 / 180.0,
        (theta - sine) / (theta_nz * two_cosine_minus_two_nz),
    )
    e = (ret_ang.view(-1, 1, 3) @ ret_lin.view(-1, 3, 1)).view(-1)

    ce_ret_ang = (c * e).view(-1, 1) * ret_ang
    jac[:, :3, 3:] = ce_ret_ang.view(-1, 3, 1) * ret_ang.view(-1, 1, 3)
    jac[:, :3, 3:] += b_ret_ang.view(-1, 3, 1) * ret_lin.view(-1, 1, 3) + ret_lin.view(
        -1, 3, 1
    ) * b_ret_ang.view(-1, 1, 3)
    diag_jac_t = torch.diagonal(jac[:, :3, 3:], dim1=1, dim2=2)
    diag_jac_t += (e * d).view(-1, 1)

    half_ret_lin = 0.5 * ret_lin
    jac[:, 0, 4] -= half_ret_lin[:, 2]
    jac[:, 1, 3] += half_ret_lin[:, 2]
    jac[:, 0, 5] += half_ret_lin[:, 1]
    jac[:, 2, 3] -= half_ret_lin[:, 1]
    jac[:, 1, 5] -= half_ret_lin[:, 0]
    jac[:, 2, 4] += half_ret_lin[:, 0]

    return [jac], torch.cat([ret_lin, ret_ang], dim=1)


class Log(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        tangent_vector = _log_impl(group)
        ctx.save_for_backward(tangent_vector, group)
        return tangent_vector

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[1]
        if not hasattr(ctx, "jacobians"):
            ctx.jacobians: torch.Tensor = _jlog_impl(group)[0][0]
            ctx.jacobians[:, :, 3:] *= 0.5

        temp = lift(
            (ctx.jacobians.transpose(1, 2) @ grad_output.unsqueeze(-1)).squeeze(-1)
        )
        jac_g = torch.einsum("nij, n...jk->n...ik", group[:, :, :3], temp)
        return jac_g


# TODO: Implement analytic backward for _jlog_impl
_log_autograd_fn = Log.apply
_jlog_autograd_fn = _jlog_impl


# -----------------------------------------------------------------------------
# Adjoint Transformation
# -----------------------------------------------------------------------------
def _adjoint_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    ret = group.new_zeros(group.shape[0], 6, 6)
    ret[:, :3, :3] = group[:, :3, :3]
    ret[:, 3:, 3:] = group[:, :3, :3]
    ret[:, :3, 3:] = SO3._hat_impl(group[:, :3, 3]) @ group[:, :3, :3]
    return ret


# NOTE: No jacobian is defined for the adjoint transformation
_jadjoint_impl = None


class Adjoint(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        ctx.save_for_backward(group)
        return _adjoint_impl(group)

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        grad_input_rot = (
            grad_output[:, :3, :3]
            + grad_output[:, 3:, 3:]
            - SO3._hat_impl(group[:, :, 3]) @ grad_output[:, :3, 3:]
        )
        grad_input_t = SO3._project_impl(
            grad_output[:, :3, 3:] @ group[:, :, :3].transpose(1, 2)
        ).view(-1, 3, 1)

        return torch.cat((grad_input_rot, grad_input_t), dim=2)


_adjoint_autograd_fn = Adjoint.apply
_jadjoint_autograd_fn = None


# -----------------------------------------------------------------------------
# Inverse
# -----------------------------------------------------------------------------
def _inverse_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    R = group[:, :, :3].transpose(1, 2)
    return torch.cat((R, -R @ group[:, :, 3:]), dim=2)


_jinverse_impl = lie_group.JInverseImplFactory(_module)


class Inverse(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        ctx.save_for_backward(group)
        return _inverse_impl(group)

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        grad_input_rot = grad_output[:, :, :3].transpose(1, 2) - group[
            :, :, 3:
        ] @ grad_output[:, :, 3:].transpose(1, 2)
        grad_input_t = -group[:, :, :3] @ grad_output[:, :, 3:]
        return torch.cat((grad_input_rot, grad_input_t), dim=2)


_inverse_autograd_fn = Inverse.apply
_jinverse_autograd_fn = _jinverse_impl


# -----------------------------------------------------------------------------
# Hat
# -----------------------------------------------------------------------------
def _hat_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    check_tangent_vector(tangent_vector)
    tangent_vector = tangent_vector.view(-1, 6)
    matrix = tangent_vector.new_zeros(tangent_vector.shape[0], 4, 4)
    matrix[:, :3, :3] = SO3._hat_impl(tangent_vector[:, 3:])
    matrix[:, :3, 3] = tangent_vector[:, :3]

    return matrix


# NOTE: No jacobian is defined for the hat operator
_jhat_impl = None


class Hat(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _hat_impl(tangent_vector)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return torch.stack(
            (
                grad_output[:, 0, 3],
                grad_output[:, 1, 3],
                grad_output[:, 2, 3],
                grad_output[:, 2, 1] - grad_output[:, 1, 2],
                grad_output[:, 0, 2] - grad_output[:, 2, 0],
                grad_output[:, 1, 0] - grad_output[:, 0, 1],
            ),
            dim=1,
        )


_hat_autograd_fn = Hat.apply
_jhat_autograd_fn = None


# -----------------------------------------------------------------------------
# Vee
# -----------------------------------------------------------------------------
def _vee_impl(matrix: torch.Tensor) -> torch.Tensor:
    check_hat_matrix(matrix)
    ret = matrix.new_zeros(matrix.shape[0], 6)
    ret[:, :3] = matrix[:, :3, 3]
    ret[:, 3:] = 0.5 * torch.stack(
        (
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ),
        dim=1,
    )
    return ret


# NOTE: No jacobian is defined for the vee operator
_jvee_impl = None


class Vee(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _vee_impl(tangent_vector)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        grad_input = grad_output.new_zeros(grad_output.shape[0], 4, 4)
        grad_input[:, :3, 3] = grad_output[:, :3]
        grad_input[:, :3, :3] = 0.5 * SO3._hat_impl(grad_output[:, 3:])
        return grad_input


_vee_autograd_fn = Vee.apply
_jvee_autograd_fn = None


# -----------------------------------------------------------------------------
# Compose
# -----------------------------------------------------------------------------
def _compose_impl(group0: torch.Tensor, group1: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group0)
    check_group_tensor(group1)
    ret_rot = group0[:, :, :3] @ group1[:, :, :3]
    ret_t = group0[:, :, :3] @ group1[:, :, 3:] + group0[:, :, 3:]
    return torch.cat((ret_rot, ret_t), dim=2)


def _jcompose_impl(
    group0: torch.Tensor, group1: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group0)
    check_group_tensor(group1)
    jacobians = []
    jacobians.append(_adjoint_autograd_fn(_inverse_autograd_fn(group1)))
    jacobians.append(group0.new_zeros(group0.shape[0], 6, 6))
    jacobians[1][:, 0, 0] = 1
    jacobians[1][:, 1, 1] = 1
    jacobians[1][:, 2, 2] = 1
    jacobians[1][:, 3, 3] = 1
    jacobians[1][:, 4, 4] = 1
    jacobians[1][:, 5, 5] = 1
    return jacobians, _compose_impl(group0, group1)


class Compose(lie_group.BinaryOperator):
    @classmethod
    def forward(cls, ctx, group0, group1):
        group0: torch.Tensor = cast(torch.Tensor, group0)
        group1: torch.Tensor = cast(torch.Tensor, group1)
        ret = _compose_impl(group0, group1)
        ctx.save_for_backward(group0, group1)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        group0: torch.Tensor = ctx.saved_tensors[0]
        group1: torch.Tensor = ctx.saved_tensors[1]
        grad_input0 = torch.cat(
            (grad_output @ group1.transpose(1, 2), grad_output[:, :, 3:]), dim=-1
        )
        grad_input1 = group0[:, :, :3].transpose(1, 2) @ grad_output
        return grad_input0, grad_input1


_compose_autograd_fn = Compose.apply
_jcompose_autograd_fn = _jcompose_impl


# -----------------------------------------------------------------------------
# Transform From
# -----------------------------------------------------------------------------
def _transform_from_impl(group: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    check_transform_tensor(tensor)
    ret = group[:, :, -1:] + group[:, :, :3] @ tensor.view(-1, 3, 1)
    return ret.reshape(tensor.shape)


def _jtransform_from_impl(
    group: torch.Tensor, tensor: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group)
    check_transform_tensor(tensor)
    jacobian_g = group.new_empty(group.shape[0], 3, 6)
    jacobian_g[:, :, :3] = SO3._hat_autograd_fn(tensor) @ group[:, :, :3]
    jacobian_g[:, :, 3:] = -group[:, :, :3] @ SO3._hat_autograd_fn(tensor)
    jacobian_p = group[:, :, :3].view(tensor.shape[:-1] + (3, 3))
    jacobians = []
    jacobians.append(jacobian_g)
    jacobians.append(jacobian_p)
    return jacobians, _transform_from_impl(group, tensor)


class TransformFrom(lie_group.BinaryOperator):
    @classmethod
    def forward(cls, ctx, group, tensor):
        group: torch.Tensor = cast(torch.Tensor, group)
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _transform_from_impl(group, tensor)
        ctx.save_for_backward(group, tensor)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        tensor: torch.Tensor = ctx.saved_tensors[1]
        grad_output: torch.Tensor = grad_output.view(-1, 3, 1)
        grad_input0 = torch.cat(
            (grad_output @ tensor.view(-1, 1, 3), grad_output), dim=-1
        )
        grad_input1 = group[:, :, :3].transpose(1, 2) @ grad_output
        return grad_input0, grad_input1.view(tensor.shape)


_transform_from_autograd_fn = TransformFrom.apply
_jtransform_from_autograd_fn = _jtransform_from_impl


# -----------------------------------------------------------------------------
# Lift
# -----------------------------------------------------------------------------
def _lift_impl(matrix: torch.Tensor) -> torch.Tensor:
    if not check_lift_matrix(matrix):
        raise ValueError("Inconsistent shape for the matrix to lift.")
    ret = matrix.new_zeros(matrix.shape[:-1] + (3, 4))
    ret[..., :, :3] = SO3._lift_impl(matrix[..., 3:])
    ret[..., :, 3] = matrix[..., :3]

    return ret


# NOTE: No jacobian is defined for the project operator
_jlift_impl = None


class Lift(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, matrix):
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _lift_impl(matrix)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return project(grad_output)


_lift_autograd_fn = Lift.apply
_jlift_autograd_fn = None

lift, jlift = lie_group.UnaryOperatorFactory(_module, "lift")


# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
def _project_impl(matrix: torch.Tensor) -> torch.Tensor:
    if not check_project_matrix(matrix):
        raise ValueError("Inconsistent shape for the matrix to project.")

    return torch.stack(
        (
            matrix[..., 0, 3],
            matrix[..., 1, 3],
            matrix[..., 2, 3],
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )


# NOTE: No jacobian is defined for the project operator
_jproject_impl = None


class Project(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, matrix):
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _project_impl(matrix)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return lift(grad_output)


_project_autograd_fn = Project.apply
_jproject_autograd_fn = None

project, jproject = lie_group.UnaryOperatorFactory(_module, "project")


# -----------------------------------------------------------------------------
# Left Act
# -----------------------------------------------------------------------------
def _left_act_impl(group: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    check_left_act_matrix(matrix)
    ret = SO3._left_act_impl(group[:, :, :3], matrix)
    return ret


def _left_act_backward_helper(group, matrix, grad_output) -> torch.Tensor:
    jac_rot = torch.einsum("n...ij,n...kj->n...ik", grad_output, matrix)
    if matrix.ndim > 3:
        dims = list(range(1, matrix.ndim - 2))
        jac_rot = jac_rot.sum(dims)
    return torch.cat((jac_rot, jac_rot.new_zeros(jac_rot.shape[0], 3, 1)), dim=-1)


class LeftAct(lie_group.BinaryOperator):
    @classmethod
    def forward(cls, ctx, group, matrix):
        group: torch.Tensor = cast(torch.Tensor, group)
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _left_act_impl(group, matrix)
        ctx.save_for_backward(group, matrix)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        group, matrix = ctx.saved_tensors
        jac_g = _left_act_backward_helper(group, matrix, grad_output)
        jac_mat = torch.einsum("nji, n...jk->n...ik", group[:, :, :3], grad_output)
        return jac_g, jac_mat


_left_act_autograd_fn = LeftAct.apply
_jleft_act_autograd_fn = None

left_act, jleft_act = lie_group.BinaryOperatorFactory(_module, "left_act")


# -----------------------------------------------------------------------------
# Left Project
# -----------------------------------------------------------------------------
_left_project_impl = lie_group.LeftProjectImplFactory(_module)
_jleft_project_impl = None


def _left_project_backward_helper(group, matrix, grad_output_lifted) -> torch.Tensor:
    group_inv = _inverse_impl(group)
    jac_ginv = _left_act_backward_helper(group_inv, matrix, grad_output_lifted)
    jac_rot = jac_ginv[:, :, :3].transpose(1, 2) - group[:, :, 3:] @ jac_ginv[
        :, :, 3:
    ].transpose(1, 2)
    jac_t = -group[:, :, :3] @ jac_ginv[:, :, 3:]
    jac_g = torch.cat((jac_rot, jac_t), dim=-1)
    return jac_g


class LeftProject(lie_group.BinaryOperator):
    @classmethod
    def forward(cls, ctx, group, matrix):
        group: torch.Tensor = cast(torch.Tensor, group)
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _left_project_impl(group, matrix)
        ctx.save_for_backward(group, matrix)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        group, matrix = ctx.saved_tensors
        grad_output_lifted = lift(grad_output)
        jac_rot = torch.einsum("n...ij,n...kj->n...ik", matrix, grad_output_lifted)
        if matrix.ndim > 3:
            dims = list(range(1, matrix.ndim - 2))
            jac_rot = jac_rot.sum(dims)
        jac_g = torch.cat((jac_rot, jac_rot.new_zeros(jac_rot.shape[0], 3, 1)), dim=-1)
        jac_mat = torch.einsum(
            "nij, n...jk->n...ik", group[:, :, :3], grad_output_lifted
        )
        return jac_g, jac_mat


_left_project_autograd_fn = LeftProject.apply
_jleft_project_autograd_fn = _jleft_project_impl

left_project, jleft_project = lie_group.BinaryOperatorFactory(_module, "left_project")

_fns = lie_group.LieGroupFns(_module)
