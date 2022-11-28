# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import cast, List, Tuple, Optional

from . import constants
from . import lie_group
from .utils import get_module


NAME: str = "SO3"
DIM: int = 3


_module = get_module(__name__)


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


def check_hat_matrix(matrix: torch.Tensor):
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError("Hat matrices of SO(3) can only be 3x3 matrices")

    if (matrix.transpose(1, 2) + matrix).abs().max().item() > constants._SO3_HAT_EPS[
        matrix.dtype
    ]:
        raise ValueError("Hat matrices of SO(3) can only be skew-symmetric.")


def check_unit_quaternion(quaternion: torch.Tensor):
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("Quaternions can only be 4-D vectors.")

    QUANTERNION_EPS = constants._SO3_QUATERNION_EPS[quaternion.dtype]

    if quaternion.dtype != torch.float64:
        quaternion = quaternion.double()

    if (torch.linalg.norm(quaternion, dim=1) - 1).abs().max().item() >= QUANTERNION_EPS:
        raise ValueError("Not unit quaternions.")


def check_left_act_matrix(matrix: torch.Tensor):
    if matrix.ndim != 3 or matrix.shape[-2] != 3:
        raise ValueError("Inconsistent shape for the matrix.")


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
    # Reference:
    # https://web.archive.org/web/20211105205926/http://planning.cs.uiuc.edu/node198.html
    if len(size) != 1:
        raise ValueError("The size should be 1D.")
    u = torch.rand(
        3,
        size[0],
        generator=generator,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    u1 = u[0]
    u2, u3 = u[1:3] * 2 * constants.PI

    a = torch.sqrt(1.0 - u1)
    b = torch.sqrt(u1)
    quaternion = torch.stack(
        [
            a * torch.sin(u2),
            a * torch.cos(u2),
            b * torch.sin(u3),
            b * torch.cos(u3),
        ],
        dim=1,
    )
    assert quaternion.shape == (size[0], 4)
    return quaternion_to_rotation(quaternion)


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
    return exp(
        constants.PI
        * torch.randn(
            size[0],
            3,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    )


# -----------------------------------------------------------------------------
# Exponential Map
# -----------------------------------------------------------------------------
def _exp_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
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


def _jexp_impl(
    tangent_vector: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
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


# TODO: Implement analytic backward for _jexp_impl
_exp_autograd_fn = Exp.apply
_jexp_autograd_fn = _jexp_impl

exp, jexp = lie_group.UnaryOperatorFactory(_module, "exp")


# -----------------------------------------------------------------------------
# Adjoint Transformation
# -----------------------------------------------------------------------------
def _adjoint_impl(group: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group):
        raise ValueError("Invalid data tensor for SO3.")
    return group


# NOTE: No jacobian is defined for the adjoint transformation
_jadjoint_impl = None


class Adjoint(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        return _adjoint_impl(group)

    @classmethod
    def backward(cls, ctx, grad_output):
        return grad_output


_adjoint_autograd_fn = Adjoint.apply
_jadjoint_autograd_fn = None

adjoint = lie_group.UnaryOperatorFactory(_module, "adjoint")


# -----------------------------------------------------------------------------
# Inverse
# -----------------------------------------------------------------------------
def _inverse_impl(group: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group):
        raise ValueError("Invalid data tensor for SO3.")
    return group.transpose(1, 2)


_jinverse_impl = lie_group.JInverseImplFactory(_module)


class Inverse(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        return _inverse_impl(group)

    @classmethod
    def backward(cls, ctx, grad_output):
        return grad_output.transpose(1, 2)


_inverse_autograd_fn = Inverse.apply
_jinverse_autograd_fn = _jinverse_impl

inverse, jinverse = lie_group.UnaryOperatorFactory(_module, "inverse")


# -----------------------------------------------------------------------------
# Hat
# -----------------------------------------------------------------------------
def _hat_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
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
    return 0.5 * torch.stack(
        (
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ),
        dim=1,
    )


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
        return 0.5 * hat(grad_output)


_vee_autograd_fn = Vee.apply
_jvee_autograd_fn = None

vee = lie_group.UnaryOperatorFactory(_module, "vee")


# -----------------------------------------------------------------------------
# Compose
# -----------------------------------------------------------------------------
def _compose_impl(group0: torch.Tensor, group1: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group0) or not check_group_tensor(group1):
        raise ValueError("Invalid data tensor for SO3.")
    return group0 @ group1


def _jcompose_impl(
    group0: torch.Tensor, group1: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if not check_group_tensor(group0) or not check_group_tensor(group1):
        raise ValueError("Invalid data tensor for SO3.")
    jacobians = []
    jacobians.append(group1.transpose(1, 2))
    jacobians.append(group0.new_zeros(group0.shape[0], 3, 3))
    jacobians[1][:, 0, 0] = 1
    jacobians[1][:, 1, 1] = 1
    jacobians[1][:, 2, 2] = 1
    return jacobians, group0 @ group1


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
        group0, group1 = ctx.saved_tensors
        return (
            grad_output @ group1.transpose(1, 2),
            group0.transpose(1, 2) @ grad_output,
        )


_compose_autograd_fn = Compose.apply
_jcompose_autograd_fn = _jcompose_impl

compose, jcompose = lie_group.BinaryOperatorFactory(_module, "compose")


# -----------------------------------------------------------------------------
# Unit Quaternion to Rotation Matrix
# -----------------------------------------------------------------------------
def _quaternion_to_rotation_impl(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    check_unit_quaternion(quaternion)

    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    q00 = w * w
    q01 = w * x
    q02 = w * y
    q03 = w * z
    q11 = x * x
    q12 = x * y
    q13 = x * z
    q22 = y * y
    q23 = y * z
    q33 = z * z

    ret = quaternion.new_zeros(quaternion.shape[0], 3, 3)
    ret[:, 0, 0] = q00 + q11 - q22 - q33
    ret[:, 0, 1] = 2 * (q12 - q03)
    ret[:, 0, 2] = 2 * (q13 + q02)
    ret[:, 1, 0] = 2 * (q12 + q03)
    ret[:, 1, 1] = q00 - q11 + q22 - q33
    ret[:, 1, 2] = 2 * (q23 - q01)
    ret[:, 2, 0] = 2 * (q13 - q02)
    ret[:, 2, 1] = 2 * (q23 + q01)
    ret[:, 2, 2] = q00 - q11 - q22 + q33
    return ret


def _jquaternion_to_rotation_impl(
    quaternion: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    check_unit_quaternion(quaternion)

    quaternion_norm = torch.norm(quaternion, dim=1, keepdim=True)
    quaternion = quaternion / quaternion_norm
    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    q00 = w * w
    q01 = w * x
    q02 = w * y
    q03 = w * z
    q11 = x * x
    q12 = x * y
    q13 = x * z
    q22 = y * y
    q23 = y * z
    q33 = z * z

    ret = quaternion.new_zeros(quaternion.shape[0], 3, 3)
    ret[:, 0, 0] = q00 + q11 - q22 - q33
    ret[:, 0, 1] = 2 * (q12 - q03)
    ret[:, 0, 2] = 2 * (q13 + q02)
    ret[:, 1, 0] = 2 * (q12 + q03)
    ret[:, 1, 1] = q00 - q11 + q22 - q33
    ret[:, 1, 2] = 2 * (q23 - q01)
    ret[:, 2, 0] = 2 * (q13 - q02)
    ret[:, 2, 1] = 2 * (q23 + q01)
    ret[:, 2, 2] = q00 - q11 - q22 + q33

    temp = -2 * quaternion / quaternion_norm
    jac = quaternion.new_zeros(quaternion.shape[0], 3, 4)
    jac[:, :, :1] = temp[:, 1:].view(-1, 3, 1)
    jac[:, :, 1:] = hat(temp[:, 1:])
    jac[:, 0, 1] = -temp[:, 0]
    jac[:, 1, 2] = -temp[:, 0]
    jac[:, 2, 3] = -temp[:, 0]

    return [jac], ret


class QuaternionToRotation(lie_group.UnaryOperator):
    @classmethod
    def forward(cls, ctx, quaternion):
        quaternion: torch.Tensor = cast(torch.Tensor, quaternion)
        ret = _quaternion_to_rotation_impl(quaternion)
        ctx.save_for_backward(quaternion, ret)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        quaternion: torch.Tensor = ctx.saved_tensors[0]
        group: torch.Tensor = ctx.saved_tensors[1]
        if not hasattr(ctx, "jacobians"):
            ctx.jacobians: torch.Tensor = _jquaternion_to_rotation_impl(quaternion)[0][
                0
            ]
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
        return grad_input.view(-1, 4)


_quaternion_to_rotation_autograd_fn = QuaternionToRotation.apply
_jquaternion_to_rotation_autograd_fn = _jquaternion_to_rotation_impl

quaternion_to_rotation, jquaternion_to_rotation = lie_group.UnaryOperatorFactory(
    _module, "quaternion_to_rotation"
)


# -----------------------------------------------------------------------------
# Lift
# -----------------------------------------------------------------------------
def _lift_impl(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-1] != 3:
        raise ValueError("Inconsistent shape for the matrix to lift.")

    ret = matrix.new_zeros(matrix.shape[:-1] + (3, 3))
    ret[..., 0, 1] = -matrix[..., 2]
    ret[..., 0, 2] = matrix[..., 1]
    ret[..., 1, 2] = -matrix[..., 0]
    ret[..., 1, 0] = matrix[..., 2]
    ret[..., 2, 0] = -matrix[..., 1]
    ret[..., 2, 1] = matrix[..., 0]

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
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return project(grad_output)


_lift_autograd_fn = Lift.apply
_jlift_autograd_fn = None

lift = lie_group.UnaryOperatorFactory(_module, "lift")


# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
def _project_impl(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Inconsistent shape for the matrix to project.")

    return torch.stack(
        (
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

project = lie_group.UnaryOperatorFactory(_module, "project")


# -----------------------------------------------------------------------------
# Left Act
# -----------------------------------------------------------------------------
def _left_act_impl(group: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if not check_group_tensor(group):
        raise ValueError("Invalid data tensor for SO3.")
    check_left_act_matrix(matrix)

    return group @ matrix


def _jleft_act_impl(
    group: torch.Tensor, matrix: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if not check_group_tensor(group):
        raise ValueError("Invalid data tensor for SO3.")
    check_left_act_matrix(matrix)
    jacobians = []
    jacobians.append(
        -torch.einsum("nij,nljk->nlik", group, lift(matrix.transpose(1, 2)))
    )
    jacobians.append(group.new_zeros(matrix.shape + matrix.shape[1:]))
    cols = torch.arange(matrix.shape[-1])
    jacobians[1][:, :, cols, :, cols] = group.view((1,) + group.shape).expand(
        matrix.shape[-1:] + group.shape
    )

    return jacobians, group @ matrix


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
        return (
            grad_output @ matrix.transpose(1, 2),
            group.transpose(1, 2) @ grad_output,
        )


_left_act_autograd_fn = LeftAct.apply
_jleft_act_autograd_fn = _jleft_act_impl

left_act, jleft_act = lie_group.BinaryOperatorFactory(_module, "left_act")
