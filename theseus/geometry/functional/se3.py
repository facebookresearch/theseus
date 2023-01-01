# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import cast, List, Tuple, Optional

from . import constants
from . import lie_group, so3
from .utils import get_module


NAME: str = "SE3"
DIM: int = 3


_module = get_module(__name__)


def check_group_tensor(tensor: torch.Tensor) -> bool:
    with torch.no_grad():
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 4):
            raise ValueError("SE3 data tensors can only be 3x4 matrices.")
    return so3.check_group_tensor(tensor[:, :, :3])


def check_tangent_vector(tangent_vector: torch.Tensor) -> bool:
    _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (6, 1)
    _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 6
    return _check


def check_hat_matrix(matrix: torch.Tensor):
    if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
        raise ValueError("Hat matrices of SE(3) can only be 3x4 matrices")

    if matrix[:, -1].abs().max() > constants._SE3_NEAR_ZERO_EPS[matrix.dtype]:
        raise ValueError("The last row for hat matrices of SE(3) must be zero")

    so3.check_hat_matrix(matrix[:, :3, :3])


# -----------------------------------------------------------------------------
# Rand
# -----------------------------------------------------------------------------
def rand(
    *size: int,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: constants.DeviceType = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    if len(size) != 1:
        raise ValueError("The size should be 1D.")
    rotation = so3.rand(
        size[0],
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    translation = torch.rand(
        size[0],
        3,
        1,
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return torch.cat((rotation, translation), dim=2)


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
    rotation = so3.randn(
        size[0],
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    translation = torch.randn(
        size[0],
        3,
        1,
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return torch.cat((rotation, translation), dim=2)


# -----------------------------------------------------------------------------
# Exponential Map
# -----------------------------------------------------------------------------
def _exp_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    if not check_tangent_vector(tangent_vector):
        raise ValueError("Tangent vectors of SE3 should be 6-D vectors.")

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
    if not check_tangent_vector(tangent_vector):
        raise ValueError("Tangent vectors of SE3 should be 6-D vectors.")

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
    jac_temp_t += so3.hat(jac_temp_v)
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

exp, jexp = lie_group.UnaryOperatorFactory(_module, "exp")


# -----------------------------------------------------------------------------
# Adjoint Transformation
# -----------------------------------------------------------------------------
def _adjoint_impl(group: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group):
        raise ValueError("Invalid data tensor for SE3.")
    ret = group.new_zeros(group.shape[0], 6, 6)
    ret[:, :3, :3] = group[:, :3, :3]
    ret[:, 3:, 3:] = group[:, :3, :3]
    ret[:, :3, 3:] = so3.hat(group[:, :3, 3]) @ group[:, :3, :3]
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
            - so3.hat(group[:, :, 3]) @ grad_output[:, :3, 3:]
        )
        grad_input_t = so3.project(
            grad_output[:, :3, 3:] @ group[:, :, :3].transpose(1, 2)
        ).view(-1, 3, 1)

        return torch.cat((grad_input_rot, grad_input_t), dim=2)


_adjoint_autograd_fn = Adjoint.apply
_jadjoint_autograd_fn = None

adjoint = lie_group.UnaryOperatorFactory(_module, "adjoint")


# -----------------------------------------------------------------------------
# Inverse
# -----------------------------------------------------------------------------
def _inverse_impl(group: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group):
        raise ValueError("Invalid data tensor for SE3.")
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

inverse, jinverse = lie_group.UnaryOperatorFactory(_module, "inverse")


# -----------------------------------------------------------------------------
# Hat
# -----------------------------------------------------------------------------
def _hat_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    if not check_tangent_vector(tangent_vector):
        raise ValueError("Tangent vectors of SE3 should be 6-D vectors.")

    matrix = tangent_vector.new_zeros(tangent_vector.shape[0], 4, 4)
    matrix[:, :3, :3] = so3.hat(tangent_vector[:, 3:])
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

hat = lie_group.UnaryOperatorFactory(_module, "hat")


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
        grad_input[:, :3, :3] = 0.5 * so3.hat(grad_output[:, 3:])
        return grad_input


_vee_autograd_fn = Vee.apply
_jvee_autograd_fn = None

vee = lie_group.UnaryOperatorFactory(_module, "vee")


# -----------------------------------------------------------------------------
# Compose
# -----------------------------------------------------------------------------
def _compose_impl(group0: torch.Tensor, group1: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group0) or not check_group_tensor(group1):
        raise ValueError("Invalid data tensor for SE3.")
    ret_rot = group0[:, :, :3] @ group1[:, :, :3]
    ret_t = group0[:, :, :3] @ group1[:, :, 3:] + group0[:, :, 3:]
    return torch.cat((ret_rot, ret_t), dim=2)


def _jcompose_impl(
    group0: torch.Tensor, group1: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if not check_group_tensor(group0) or not check_group_tensor(group1):
        raise ValueError("Invalid data tensor for SE3.")
    jacobians = []
    jacobians.append(adjoint(inverse(group1)))
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
        grad_input0 = group0.new_zeros(grad_output.shape[0], 3, 4)
        grad_input0[:, :, :3] = grad_output @ group1.transpose(1, 2)
        grad_input0[:, :, 3] = grad_output[:, :, 3]
        grad_input1 = group0[:, :, :3].transpose(1, 2) @ grad_output
        return grad_input0, grad_input1


_compose_autograd_fn = Compose.apply
_jcompose_autograd_fn = _jcompose_impl

compose, jcompose = lie_group.BinaryOperatorFactory(_module, "compose")
