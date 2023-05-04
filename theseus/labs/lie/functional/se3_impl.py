# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import cast, List, Tuple, Optional

from . import constants
from . import lie_group, so3_impl as SO3
from .check_contexts import checks_base
from .utils import get_module, shape_err_msg


NAME: str = "SE3"
DIM: int = 6


_module = get_module(__name__)


def check_group_tensor(tensor: torch.Tensor):
    def _impl(t_: torch.Tensor):
        SO3.check_group_tensor(t_[..., :3])

    if tensor.shape[-2:] != (3, 4):
        raise ValueError(shape_err_msg("SE3 data tensors", "(..., 3, 4)", tensor.shape))

    checks_base(tensor, _impl)


def check_matrix_tensor(tensor: torch.Tensor):
    if tensor.shape[-2:] != (3, 4):
        raise ValueError(shape_err_msg("SE3 data tensors", "(..., 3, 4)", tensor.shape))


def check_transform_tensor(tensor: torch.Tensor):
    SO3.check_transform_tensor(tensor)


def check_tangent_vector(tangent_vector: torch.Tensor):
    _check = tangent_vector.shape[-2:] == (6, 1)
    _check |= tangent_vector.shape[-1] == 6
    if not _check:
        raise ValueError(
            shape_err_msg(
                "Tangent vectors of SE3",
                "(..., 6) or (..., 6, 1)",
                tangent_vector.shape,
            )
        )


def check_hat_matrix(matrix: torch.Tensor):
    def _impl(t_: torch.Tensor):
        if t_[..., -1].abs().max() > constants._SE3_NEAR_ZERO_EPS[t_.dtype]:
            raise ValueError("The last row for hat matrices of SE3 must be zero")

        SO3.check_hat_matrix(t_[..., :3, :3])

    if matrix.shape[-2:] != (4, 4):
        raise ValueError(
            shape_err_msg("Hat matrices of SE3", "(..., 4, 4)", matrix.shape)
        )

    checks_base(matrix, _impl)


def check_lift_matrix(matrix: torch.Tensor):
    if not matrix.shape[-1] == 6:
        raise ValueError(
            shape_err_msg("Lifted matrices of SE3", "(..., 6)", matrix.shape)
        )


def check_project_matrix(matrix: torch.Tensor):
    if not matrix.shape[-2:] == (3, 4):
        raise ValueError(
            shape_err_msg("Projected matrices of SE3", "(..., 3, 4)", matrix.shape)
        )


def check_left_act_matrix(matrix: torch.Tensor):
    if matrix.shape[-2] != 3:
        raise ValueError(
            shape_err_msg("Left acted matrices of SE3", "(..., 3, -1)", matrix.shape)
        )


def check_left_project_matrix(matrix: torch.Tensor):
    if matrix.shape[-2:] != (3, 4):
        raise ValueError(
            shape_err_msg("Left projected matrices of SE3", "(..., 3, 4)", matrix.shape)
        )


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
    rotation = SO3.rand(*size, generator=generator, dtype=dtype, device=device)
    translation = torch.rand(
        *size, 3, 1, generator=generator, dtype=dtype, device=device
    )
    ret = torch.cat((rotation, translation), dim=-1)
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
    rotation = SO3.randn(*size, generator=generator, dtype=dtype, device=device)
    translation = torch.randn(
        *size, 3, 1, generator=generator, dtype=dtype, device=device
    )
    ret = torch.cat((rotation, translation), dim=-1)
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
    ret = torch.tensor(_BASE_IDENTITY_SE3, dtype=dtype, device=device).repeat(
        *size, 1, 1
    )
    ret.requires_grad_(requires_grad)
    return ret


# -----------------------------------------------------------------------------
# Exponential Map
# -----------------------------------------------------------------------------
def _exp_impl_helper(tangent_vector: torch.Tensor):
    size = tangent_vector.shape[:-1]
    ret = tangent_vector.new_zeros(*size, 3, 4)

    # Compute the rotation
    ret[..., :3], (
        theta,
        theta2,
        theta_nz,
        theta2_nz,
        sine,
        _,
        sine_by_theta,
        one_minus_cosine_by_theta2,
    ) = SO3._exp_impl_helper(tangent_vector[..., 3:])

    # Compute the translation
    tangent_vector_lin = tangent_vector[..., :3].view(*size, 3, 1)
    tangent_vector_ang = tangent_vector[..., 3:].view(*size, 3, 1)
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
    theta3_nz = theta_nz * theta2_nz
    theta_minus_sine_by_theta3_t = torch.where(
        near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
    )
    ret[..., 3:] = sine_by_theta * tangent_vector_lin
    ret[..., 3:] += one_minus_cosine_by_theta2 * torch.cross(
        tangent_vector_ang, tangent_vector_lin, dim=-2
    )
    ret[..., 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(-1, -2) @ tangent_vector_lin)
    )

    return ret, (
        theta,
        theta2_nz,
        sine_by_theta,
        one_minus_cosine_by_theta2,
        theta_minus_sine_by_theta3_t,
    )


def _exp_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    check_tangent_vector(tangent_vector)
    ret, _ = _exp_impl_helper(tangent_vector)
    return ret


def _jexp_impl_helper(
    tangent_vector: torch.Tensor,
    rotation: torch.Tensor,
    theta: torch.Tensor,
    theta2_nz: torch.Tensor,
    sine_by_theta: torch.Tensor,
    one_minus_cosine_by_theta2: torch.Tensor,
    theta_minus_sine_by_theta3_t: torch.Tensor,
    theta_minus_sine_by_theta3_rot: torch.Tensor,
):
    size = tangent_vector.shape[:-1]
    jac = tangent_vector.new_zeros(*size, 6, 6)

    # compute rotation jacobians
    jac[..., :3, :3], _ = SO3._jexp_impl_helper(
        tangent_vector[..., 3:],
        sine_by_theta,
        one_minus_cosine_by_theta2,
        theta_minus_sine_by_theta3_rot,
    )
    jac[..., 3:, 3:] = jac[..., :3, :3]

    # compute translation jacobians
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
    d_one_minus_cosine_by_theta2 = torch.where(
        near_zero,
        constants._NEAR_ZERO_D_ONE_MINUS_COSINE_BY_THETA2,
        (sine_by_theta - 2 * one_minus_cosine_by_theta2) / theta2_nz,
    )
    d_theta_minus_sine_by_theta3 = torch.where(
        near_zero,
        constants._NEAR_ZERO_D_THETA_MINUS_SINE_BY_THETA3,
        (one_minus_cosine_by_theta2 - 3 * theta_minus_sine_by_theta3_t) / theta2_nz,
    )

    w = tangent_vector[..., 3:]
    v = tangent_vector[..., :3]
    wv = w.cross(v, dim=-1)
    wwv = w.cross(wv, dim=-1)
    sw = theta_minus_sine_by_theta3_t.view(*size, 1) * w

    jac_temp_t = (
        d_one_minus_cosine_by_theta2.view(*size, 1) * wv
        + d_theta_minus_sine_by_theta3.view(*size, 1) * wwv
    ).view(*size, 3, 1) @ w.view(*size, 1, 3)
    jac_temp_t -= v.view(*size, 3, 1) @ sw.view(*size, 1, 3)
    jac_temp_v = (
        -one_minus_cosine_by_theta2.view(*size, 1) * v
        - theta_minus_sine_by_theta3_t.view(*size, 1) * wv
    )
    jac_temp_t += SO3._hat_autograd_fn(jac_temp_v)
    diag_jac_t = torch.diagonal(jac_temp_t, dim1=-1, dim2=-2)
    diag_jac_t += (sw.view(*size, 1, 3) @ v.view(*size, 3, 1)).view(*size, 1)

    jac[..., :3, 3:] = rotation.transpose(-1, -2) @ jac_temp_t

    return jac, (None,)


def _jexp_impl(
    tangent_vector: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_tangent_vector(tangent_vector)
    ret, (
        theta,
        theta2_nz,
        sine_by_theta,
        one_minus_cosine_by_theta2,
        theta_minus_sine_by_theta3_t,
    ) = _exp_impl_helper(tangent_vector)
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
    theta_minus_sine_by_theta3_rot = torch.where(
        near_zero, torch.zeros_like(theta), theta_minus_sine_by_theta3_t
    )
    jac, _ = _jexp_impl_helper(
        tangent_vector,
        ret[..., :3],
        theta,
        theta2_nz,
        sine_by_theta,
        one_minus_cosine_by_theta2,
        theta_minus_sine_by_theta3_t,
        theta_minus_sine_by_theta3_rot,
    )

    return [jac], ret


class Exp(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _exp_impl(tangent_vector)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (tangent_vector,). outputs is exp_map
        ctx.save_for_backward(inputs[0], outputs)

    @classmethod
    def backward(cls, ctx, grad_output):
        tangent_vector: torch.Tensor = ctx.saved_tensors[0]
        group: torch.Tensor = ctx.saved_tensors[1]
        jacs = _jexp_impl(tangent_vector)[0][0]
        size = (
            tangent_vector.shape[:-2]
            if tangent_vector.shape[-1] == 1
            else tangent_vector.shape[:-1]
        )
        dg = group[..., :3].transpose(-2, -1) @ grad_output
        grad_input = jacs.transpose(-2, -1) @ torch.stack(
            (
                dg[..., 0, 3],
                dg[..., 1, 3],
                dg[..., 2, 3],
                dg[..., 2, 1] - dg[..., 1, 2],
                dg[..., 0, 2] - dg[..., 2, 0],
                dg[..., 1, 0] - dg[..., 0, 1],
            ),
            dim=-1,
        ).view(*size, 6, 1)
        return grad_input.view_as(tangent_vector)


# TODO: Implement analytic backward for _jexp_impl
_exp_autograd_fn = Exp.apply
_jexp_autograd_fn = _jexp_impl


# -----------------------------------------------------------------------------
# Logarithm Map
# -----------------------------------------------------------------------------
def _log_impl_helper(group: torch.Tensor):
    check_group_tensor(group)
    size = group.shape[:-2]

    # Compute the rotation
    ret_ang, (theta, sine, cosine) = SO3._log_impl_helper(group[..., :3])

    # Compute the translation
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[group.dtype]
    theta2 = theta**2
    sine_theta = sine * theta
    two_cosine_minus_two = 2 * cosine - 2
    two_cosine_minus_two_nz = torch.where(
        near_zero, constants._NON_ZERO, two_cosine_minus_two
    )

    theta2_nz = torch.where(near_zero, constants._NON_ZERO, theta2)

    a = torch.where(near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz)
    b = torch.where(
        near_zero,
        1.0 / 12 + theta2 / 720,
        (sine_theta + two_cosine_minus_two) / (theta2_nz * two_cosine_minus_two_nz),
    )

    translation = group[..., 3].view(*size, 3, 1)
    ret_lin = a.view(*size, 1) * group[..., 3]
    ret_lin -= 0.5 * torch.cross(ret_ang, group[..., 3], dim=-1)
    ret_ang_ext = ret_ang.view(*size, 3, 1)
    ret_lin += b.view(*size, 1) * (
        ret_ang_ext @ (ret_ang_ext.transpose(-1, -2) @ translation)
    ).view(*size, 3)

    return torch.cat([ret_lin, ret_ang], dim=-1), (
        theta,
        theta2,
        theta2_nz,
        sine,
        cosine,
        two_cosine_minus_two_nz,
        a,
        b,
    )


def _log_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    ret, _ = _log_impl_helper(group)
    return ret


def _jlog_impl_helper(
    tangent_vector: torch.Tensor,
    theta: torch.Tensor,
    theta2: torch.Tensor,
    theta2_nz: torch.Tensor,
    sine: torch.Tensor,
    cosine: torch.Tensor,
    two_cosine_minus_two_nz: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    ret_lin = tangent_vector[..., :3]
    ret_ang = tangent_vector[..., 3:]
    size = tangent_vector.shape[:-1]
    near_zero = theta < constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
    jac = tangent_vector.new_zeros(*size, 6, 6)
    jac[..., :3, :3], (b_ret_ang,) = SO3._jlog_impl_helper(ret_ang, theta, sine, cosine)
    jac[..., 3:, 3:] = jac[..., :3, :3]

    theta_nz = torch.where(near_zero, constants._NON_ZERO, theta)
    theta4_nz = theta2_nz**2
    c = torch.where(
        near_zero,
        -1 / 360.0 - theta2 / 7560.0,
        -(2 * two_cosine_minus_two_nz + theta * sine + theta2)
        / (theta4_nz * two_cosine_minus_two_nz),
    )
    d = torch.where(
        near_zero,
        -1 / 6.0 - theta2 / 180.0,
        (theta - sine) / (theta_nz * two_cosine_minus_two_nz),
    )
    e = (ret_ang.view(*size, 1, 3) @ ret_lin.view(*size, 3, 1)).view(*size)

    ce_ret_ang = (c * e).view(*size, 1) * ret_ang
    jac[..., :3, 3:] = ce_ret_ang.view(*size, 3, 1) * ret_ang.view(*size, 1, 3)
    jac[..., :3, 3:] += b_ret_ang.view(*size, 3, 1) * ret_lin.view(
        *size, 1, 3
    ) + ret_lin.view(*size, 3, 1) * b_ret_ang.view(*size, 1, 3)
    diag_jac_t = torch.diagonal(jac[..., :3, 3:], dim1=-1, dim2=-2)
    diag_jac_t += (e * d).view(*size, 1)

    half_ret_lin = 0.5 * ret_lin
    jac[..., 0, 4] -= half_ret_lin[..., 2]
    jac[..., 1, 3] += half_ret_lin[..., 2]
    jac[..., 0, 5] += half_ret_lin[..., 1]
    jac[..., 2, 3] -= half_ret_lin[..., 1]
    jac[..., 1, 5] -= half_ret_lin[..., 0]
    jac[..., 2, 4] += half_ret_lin[..., 0]

    return jac, (None,)


def _jlog_impl(group: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group)
    tangent_vector, (
        theta,
        theta2,
        theta2_nz,
        sine,
        cosine,
        two_cosine_minus_two_nz,
        a,
        b,
    ) = _log_impl_helper(group)
    jac, _ = _jlog_impl_helper(
        tangent_vector,
        theta,
        theta2,
        theta2_nz,
        sine,
        cosine,
        two_cosine_minus_two_nz,
        a,
        b,
    )

    return [jac], tangent_vector


class Log(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        tangent_vector = _log_impl(group)
        return tangent_vector

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group,). outputs is tangent_vector
        ctx.save_for_backward(outputs, inputs[0])

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[1]
        jacobians = _jlog_impl(group)[0][0]
        jacobians[..., 3:] *= 0.5
        temp = lift(
            (jacobians.transpose(-1, -2) @ grad_output.unsqueeze(-1)).squeeze(-1)
        )

        jac_g = torch.einsum("n...ij, n...jk->n...ik", group[..., :3], temp)
        return jac_g


# TODO: Implement analytic backward for _jlog_impl
_log_autograd_fn = Log.apply
_jlog_autograd_fn = _jlog_impl


# -----------------------------------------------------------------------------
# Adjoint Transformation
# -----------------------------------------------------------------------------
def _adjoint_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    size = group.shape[:-2]
    ret = group.new_zeros(*size, 6, 6)
    ret[..., :3, :3] = group[..., :3, :3]
    ret[..., 3:, 3:] = group[..., :3, :3]
    ret[..., :3, 3:] = SO3._hat_impl(group[..., :3, 3]) @ group[..., :3, :3]
    return ret


# NOTE: No jacobian is defined for the adjoint transformation
_jadjoint_impl = None


class Adjoint(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        return _adjoint_impl(group)

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0])  # inputs is (group,)

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        size = group.shape[:-2]
        grad_input_rot = (
            grad_output[..., :3, :3]
            + grad_output[..., 3:, 3:]
            - SO3._hat_impl(group[..., 3]) @ grad_output[..., :3, 3:]
        )
        grad_input_t = SO3._project_impl(
            grad_output[..., :3, 3:] @ group[..., :3].transpose(-2, -1)
        ).view(*size, 3, 1)

        return torch.cat((grad_input_rot, grad_input_t), dim=-1)


_adjoint_autograd_fn = Adjoint.apply
_jadjoint_autograd_fn = None


# -----------------------------------------------------------------------------
# Inverse
# -----------------------------------------------------------------------------
def _inverse_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    R = group[..., :3].transpose(-2, -1)
    return torch.cat((R, -R @ group[..., 3:]), dim=-1)


_jinverse_impl = lie_group.JInverseImplFactory(_module)


class Inverse(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        return _inverse_impl(group)

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0])  # inputs is (group,)

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        grad_input_rot = grad_output[..., :3].transpose(-1, -2) - group[
            ..., 3:
        ] @ grad_output[..., 3:].transpose(-1, -2)
        grad_input_t = -group[..., :3] @ grad_output[..., 3:]
        return torch.cat((grad_input_rot, grad_input_t), dim=-1)


_inverse_autograd_fn = Inverse.apply
_jinverse_autograd_fn = _jinverse_impl


# -----------------------------------------------------------------------------
# Hat
# -----------------------------------------------------------------------------
def _hat_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    check_tangent_vector(tangent_vector)
    size = (
        tangent_vector.shape[:-2]
        if tangent_vector.shape[-1] == 1
        else tangent_vector.shape[:-1]
    )
    tangent_vector = tangent_vector.view(*size, 6)
    matrix = tangent_vector.new_zeros(*size, 4, 4)
    matrix[..., :3, :3] = SO3._hat_impl(tangent_vector[..., 3:])
    matrix[..., :3, 3] = tangent_vector[..., :3]

    return matrix


# NOTE: No jacobian is defined for the hat operator
_jhat_impl = None


class Hat(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _hat_impl(tangent_vector)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (tangent_vector, )
        ctx.save_for_backward(inputs[0])

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        tangent_vector: torch.Tensor = ctx.saved_tensors[0]
        return torch.stack(
            (
                grad_output[..., 0, 3],
                grad_output[..., 1, 3],
                grad_output[..., 2, 3],
                grad_output[..., 2, 1] - grad_output[..., 1, 2],
                grad_output[..., 0, 2] - grad_output[..., 2, 0],
                grad_output[..., 1, 0] - grad_output[..., 0, 1],
            ),
            dim=-1,
        ).view_as(tangent_vector)


_hat_autograd_fn = Hat.apply
_jhat_autograd_fn = None


# -----------------------------------------------------------------------------
# Vee
# -----------------------------------------------------------------------------
def _vee_impl(matrix: torch.Tensor) -> torch.Tensor:
    check_hat_matrix(matrix)
    size = matrix.shape[:-2]
    ret = matrix.new_zeros(*size, 6)
    ret[..., :3] = matrix[..., :3, 3]
    ret[..., 3:] = 0.5 * torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    return ret


# NOTE: No jacobian is defined for the vee operator
_jvee_impl = None


class Vee(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _vee_impl(tangent_vector)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        pass

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        size = grad_output.shape[:-1]
        grad_input = grad_output.new_zeros(*size, 4, 4)
        grad_input[..., :3, 3] = grad_output[..., :3]
        grad_input[..., :3, :3] = 0.5 * SO3._hat_impl(grad_output[..., 3:])
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
    def _forward_impl(cls, group0, group1):
        group0: torch.Tensor = cast(torch.Tensor, group0)
        group1: torch.Tensor = cast(torch.Tensor, group1)
        ret = _compose_impl(group0, group1)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0], inputs[1])  # the two groups

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
    jacobian_g[:, :, :3] = group[:, :, :3]
    jacobian_g[:, :, 3:] = -group[:, :, :3] @ SO3._hat_autograd_fn(tensor)
    jacobian_p = group[:, :, :3].view(tensor.shape[:-1] + (3, 3))
    jacobians = []
    jacobians.append(jacobian_g)
    jacobians.append(jacobian_p)
    return jacobians, _transform_from_impl(group, tensor)


class TransformFrom(lie_group.BinaryOperator):
    @classmethod
    def _forward_impl(cls, group, tensor):
        group: torch.Tensor = cast(torch.Tensor, group)
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _transform_from_impl(group, tensor)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group, tensor)
        ctx.save_for_backward(inputs[0], inputs[1])

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
    check_lift_matrix(matrix)
    ret = matrix.new_zeros(matrix.shape[:-1] + (3, 4))
    ret[..., :3] = SO3._lift_impl(matrix[..., 3:])
    ret[..., 3] = matrix[..., :3]

    return ret


# NOTE: No jacobian is defined for the project operator
_jlift_impl = None


class Lift(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, matrix):
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _lift_impl(matrix)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        pass

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
    check_project_matrix(matrix)
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
    def _forward_impl(cls, matrix):
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _project_impl(matrix)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        pass

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
    ret = SO3._left_act_impl(group[..., :3], matrix)
    return ret


def _left_act_backward_helper(group, matrix, grad_output) -> torch.Tensor:
    jac_rot = torch.einsum("n...ij,n...kj->n...ik", grad_output, matrix)
    if matrix.ndim > 3:
        dims = list(range(1, matrix.ndim - 2))
        jac_rot = jac_rot.sum(dims)
    return torch.cat((jac_rot, jac_rot.new_zeros(jac_rot.shape[0], 3, 1)), dim=-1)


class LeftAct(lie_group.BinaryOperator):
    @classmethod
    def _forward_impl(cls, group, matrix):
        group: torch.Tensor = cast(torch.Tensor, group)
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _left_act_impl(group, matrix)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0], inputs[1])

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


class LeftProject(lie_group.BinaryOperator):
    @classmethod
    def _forward_impl(cls, group, matrix):
        group: torch.Tensor = cast(torch.Tensor, group)
        matrix: torch.Tensor = cast(torch.Tensor, matrix)
        ret = _left_project_impl(group, matrix)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0], inputs[1])

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


# -----------------------------------------------------------------------------
# Normalize
# -----------------------------------------------------------------------------
def _normalize_impl(matrix: torch.Tensor) -> torch.Tensor:
    check_matrix_tensor(matrix)
    rotation = SO3._normalize_impl_helper(matrix[..., :3])[0]
    translation = matrix[..., 3:]
    return torch.cat((rotation, translation), dim=-1)


class Normalize(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, matrix):
        check_matrix_tensor(matrix)
        matrix: torch.Tensor = matrix
        rotation, svd_info = SO3._normalize_impl_helper(matrix[..., :3])
        translation = matrix[..., 3:]
        output = torch.cat((rotation, translation), dim=-1)
        return output, svd_info

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # outputs is (normalized_out, svd_info)
        svd_info = outputs[1]
        ctx.save_for_backward(
            svd_info["u"], svd_info["s"], svd_info["v"], svd_info["sign"]
        )

    @classmethod
    def backward(cls, ctx, grad_output, _):
        u, s, v, sign = ctx.saved_tensors
        grad_input1 = SO3._normalize_backward_helper(
            u, s, v, sign, grad_output[..., :3]
        )
        grad_input2 = grad_output[..., 3:]
        grad_input = torch.cat((grad_input1, grad_input2), dim=-1)
        return grad_input, None


def _normalize_autograd_fn(matrix: torch.Tensor):
    return Normalize.apply(matrix)[0]


_jnormalize_autograd_fn = None


_fns = lie_group.LieGroupFns(_module)
