# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, cast

import torch

from torchlie.global_params import _TORCHLIE_GLOBAL_PARAMS as LIE_PARAMS

from . import constants, lie_group
from .check_contexts import checks_base
from .utils import (
    fill_dims,
    get_module,
    permute_op_dim,
    shape_err_msg,
    unpermute_op_dim,
)

NAME: str = "SO3"
DIM: int = 3

_DIAG_3_IDX = [0, 1, 2]

_module = get_module(__name__)


def check_group_tensor(tensor: torch.Tensor):
    def _impl(t_):
        MATRIX_EPS = LIE_PARAMS.get_eps("so3", "matrix", t_.dtype)
        if t_.dtype != torch.float64:
            t_ = t_.double()

        _check = (
            torch.matmul(t_, t_.transpose(-1, -2))
            - torch.eye(3, 3, dtype=t_.dtype, device=t_.device)
        ).abs().max().item() < MATRIX_EPS
        _check &= (torch.linalg.det(t_) - 1).abs().max().item() < MATRIX_EPS

        if not _check:
            raise ValueError("Invalid data tensor for SO3.")

    if tensor.shape[-2:] != (3, 3):
        raise ValueError(shape_err_msg("SO3 data tensors", "(..., 3, 3)", tensor.shape))

    checks_base(tensor, _impl)


def check_group_shape(tensor: torch.Tensor):
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(shape_err_msg("SO3 data tensors", "(..., 3, 3)", tensor.shape))


def check_tangent_vector(tangent_vector: torch.Tensor):
    _check = tangent_vector.shape[-1] == 3
    if not _check:
        raise ValueError(
            shape_err_msg(
                "Tangent vectors of SO3",
                "(..., 3)",
                tangent_vector.shape,
            )
        )


def check_hat_tensor(tensor: torch.Tensor):
    def _impl(t_: torch.Tensor):
        if (t_.transpose(-1, -2) + t_).abs().max().item() > LIE_PARAMS.get_eps(
            "so3", "hat", t_.dtype
        ):
            raise ValueError("Hat tensors of SO3 can only be skew-symmetric.")

    if tensor.shape[-2:] != (3, 3):
        raise ValueError(
            shape_err_msg("Hat tensors of SO3", "(..., 3, 3)", tensor.shape)
        )

    checks_base(tensor, _impl)


def check_transform_tensor(tensor: torch.Tensor):
    if tensor.shape[-1] != 3:
        raise ValueError(
            shape_err_msg(
                "Tensors transformed by SO3",
                "(..., 3)",
                tensor.shape,
            )
        )


def check_lift_tensor(tensor: torch.Tensor):
    if not tensor.shape[-1] == 3:
        raise ValueError(
            shape_err_msg("Lifted tensors of SO3", "(..., 3)", tensor.shape)
        )


def check_project_tensor(tensor: torch.Tensor):
    if not tensor.shape[-2:] == (3, 3):
        raise ValueError(
            shape_err_msg("Projected tensors of SO3", "(..., 3, 3)", tensor.shape)
        )


def check_unit_quaternion(quaternion: torch.Tensor):
    def _impl(t_: torch.Tensor):
        QUANTERNION_EPS = LIE_PARAMS.get_eps("so3", "quat", t_.dtype)

        if t_.dtype != torch.float64:
            t_ = t_.double()

        if (torch.linalg.norm(t_, dim=-1) - 1).abs().max().item() >= QUANTERNION_EPS:
            raise ValueError("Not unit quaternions.")

    if quaternion.shape[-1] != 4:
        raise ValueError(shape_err_msg("Quaternions", "(..., 4)", quaternion.shape))

    checks_base(quaternion, _impl)


def check_left_act_tensor(tensor: torch.Tensor):
    if tensor.shape[-2] != 3:
        raise ValueError(
            shape_err_msg("Left acted tensors of SO3", "(..., 3, -1)", tensor.shape)
        )


def check_left_project_tensor(tensor: torch.Tensor):
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(
            shape_err_msg("Left projected tensors of SO3", "(..., 3, 3)", tensor.shape)
        )


def get_group_size(group: torch.Tensor):
    return group.shape[:-2]


def get_tangent_vector_size(tangent_vector: torch.Tensor):
    return tangent_vector.shape[:-1]


def get_transform_tensor_size(transform_tensor: torch.Tensor):
    return transform_tensor.shape[:-1]


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
    u = torch.rand(3, *size, generator=generator, dtype=dtype, device=device)
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
        dim=-1,
    )
    assert quaternion.shape == size + (4,)
    ret = _quaternion_to_rotation_autograd_fn(quaternion)
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
    ret = _exp_autograd_fn(
        constants.PI
        * torch.randn(*size, 3, generator=generator, dtype=dtype, device=device)
    )
    ret.requires_grad_(requires_grad)
    return ret


# -----------------------------------------------------------------------------
# Identity
# -----------------------------------------------------------------------------
def identity(
    *size: int,
    dtype: Optional[torch.dtype] = None,
    device: constants.DeviceType = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    ret = torch.eye(3, device=device, dtype=dtype).repeat(*size, 1, 1)
    ret.requires_grad_(requires_grad)
    return ret


# -----------------------------------------------------------------------------
# Exponential Map
# -----------------------------------------------------------------------------
def _exp_impl_helper(tangent_vector: torch.Tensor):
    theta = torch.linalg.norm(tangent_vector, dim=-1, keepdim=True).unsqueeze(-1)
    theta2 = theta**2
    # Compute the approximations when theta ~ 0
    near_zero = theta < LIE_PARAMS.get_eps("so3", "near_zero", tangent_vector.dtype)
    theta_nz = torch.where(near_zero, constants._NON_ZERO, theta)
    theta2_nz = torch.where(near_zero, constants._NON_ZERO, theta2)

    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine = theta.sin()
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(
        near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
    )

    size = tangent_vector.shape[:-1]
    ret = (
        one_minus_cosine_by_theta2
        * tangent_vector.view(*size, 3, 1)
        @ tangent_vector.view(*size, 1, 3)
    )
    ret[..., 0, 0] += cosine.view(size)
    ret[..., 1, 1] += cosine.view(size)
    ret[..., 2, 2] += cosine.view(size)
    sine_axis = sine_by_theta.view(*size, 1) * tangent_vector
    ret[..., 0, 1] -= sine_axis[..., 2]
    ret[..., 1, 0] += sine_axis[..., 2]
    ret[..., 0, 2] += sine_axis[..., 1]
    ret[..., 2, 0] -= sine_axis[..., 1]
    ret[..., 1, 2] -= sine_axis[..., 0]
    ret[..., 2, 1] += sine_axis[..., 0]

    return ret, (
        theta,
        theta2,
        theta_nz,
        theta2_nz,
        sine,
        cosine,
        sine_by_theta,
        one_minus_cosine_by_theta2,
    )


def _exp_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    check_tangent_vector(tangent_vector)
    ret, _ = _exp_impl_helper(tangent_vector)
    return ret


def _jexp_impl_helper(
    tangent_vector: torch.Tensor,
    sine_by_theta: torch.Tensor,
    one_minus_cosine_by_theta2: torch.Tensor,
    theta_minus_sine_by_theta3: torch.Tensor,
):
    size = tangent_vector.shape[:-1]
    jac = theta_minus_sine_by_theta3 * (
        tangent_vector.view(*size, 3, 1) @ tangent_vector.view(*size, 1, 3)
    )
    diag_jac = jac.diagonal(dim1=-1, dim2=-2)
    diag_jac += sine_by_theta.view(*size, 1)
    jac_temp = one_minus_cosine_by_theta2.view(*size, 1) * tangent_vector
    jac[..., 0, 1] += jac_temp[..., 2]
    jac[..., 1, 0] -= jac_temp[..., 2]
    jac[..., 0, 2] -= jac_temp[..., 1]
    jac[..., 2, 0] += jac_temp[..., 1]
    jac[..., 1, 2] += jac_temp[..., 0]
    jac[..., 2, 1] -= jac_temp[..., 0]

    return jac, (None,)


def _jexp_impl(
    tangent_vector: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_tangent_vector(tangent_vector)
    ret, (
        theta,
        _,
        theta_nz,
        theta2_nz,
        sine,
        _,
        sine_by_theta,
        one_minus_cosine_by_theta2,
    ) = _exp_impl_helper(tangent_vector)

    near_zero = theta < LIE_PARAMS.get_eps("so3", "near_zero", tangent_vector.dtype)
    theta3_nz = theta_nz * theta2_nz
    theta_minus_sine_by_theta3 = torch.where(
        near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
    )
    jac, _ = _jexp_impl_helper(
        tangent_vector,
        sine_by_theta,
        one_minus_cosine_by_theta2,
        theta_minus_sine_by_theta3,
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
        # inputs is (tangent_vector, ). outputs is exp_map
        ctx.save_for_backward(inputs[0], outputs)

    @classmethod
    def backward(cls, ctx, grad_output):
        tangent_vector: torch.Tensor = ctx.saved_tensors[0]
        group: torch.Tensor = ctx.saved_tensors[1]
        jacs = _jexp_impl(tangent_vector)[0][0]
        dR = group.transpose(-2, -1) @ grad_output
        size = get_tangent_vector_size(tangent_vector)
        grad_input = jacs.transpose(-2, -1) @ torch.stack(
            (
                dR[..., 2, 1] - dR[..., 1, 2],
                dR[..., 0, 2] - dR[..., 2, 0],
                dR[..., 1, 0] - dR[..., 0, 1],
            ),
            dim=-1,
        ).view(*size, 3, 1)
        return grad_input.view_as(tangent_vector)


# TODO: Implement analytic backward for _jexp_impl
_exp_autograd_fn = Exp.apply
_jexp_autograd_fn = _jexp_impl


_UPPER_IDX_3x3_CUDA: Dict[str, torch.Tensor] = None
if torch.cuda.is_available():
    _UPPER_IDX_3x3_CUDA = {
        f"cuda:{i}": torch.triu_indices(3, 3, offset=1).to(f"cuda:{i}").flip(-1)
        for i in range(torch.cuda.device_count())
    }


def _sine_axis_fn(group: torch.Tensor, size: torch.Size) -> torch.Tensor:
    if LIE_PARAMS._faster_log_maps:
        if group.is_cuda:
            g_minus_gt = 0.5 * (group.adjoint() - group)
            upper_idx = _UPPER_IDX_3x3_CUDA[str(group.device)]
            sine_axis = g_minus_gt[..., upper_idx[0], upper_idx[1]]
            sine_axis[..., 1] *= -1
        else:
            sine_axis = group.new_zeros(*size, 3)
            sine_axis[..., 0] = group[..., 2, 1] - group[..., 1, 2]
            sine_axis[..., 1] = group[..., 0, 2] - group[..., 2, 0]
            sine_axis[..., 2] = group[..., 1, 0] - group[..., 0, 1]
            sine_axis *= 0.5
    else:
        sine_axis = group.new_zeros(*size, 3)
        sine_axis[..., 0] = 0.5 * (group[..., 2, 1] - group[..., 1, 2])
        sine_axis[..., 1] = 0.5 * (group[..., 0, 2] - group[..., 2, 0])
        sine_axis[..., 2] = 0.5 * (group[..., 1, 0] - group[..., 0, 1])
    return sine_axis


# -----------------------------------------------------------------------------
# Logarithm Map
# -----------------------------------------------------------------------------
def _log_impl_helper(group: torch.Tensor):
    size = get_group_size(group)
    sine_axis = _sine_axis_fn(group, size)
    cosine = 0.5 * (group.diagonal(dim1=-1, dim2=-2).sum(dim=-1) - 1)
    sine = sine_axis.norm(dim=-1)
    theta = torch.atan2(sine, cosine)

    near_zero = theta < LIE_PARAMS.get_eps("so3", "near_zero", group.dtype)
    near_pi = 1 + cosine <= LIE_PARAMS.get_eps("so3", "near_pi", group.dtype)
    # theta != pi
    near_zero_or_near_pi = torch.logical_or(near_zero, near_pi)
    # Compute the approximation of theta / sin(theta) when theta is near to 0
    sine_nz = torch.where(near_zero_or_near_pi, constants._NON_ZERO, sine)
    scale = torch.where(
        near_zero_or_near_pi,
        1 + sine**2 / 6,
        theta / sine_nz,
    )
    ret = sine_axis * scale.view(*size, 1)

    # # theta ~ pi
    ddiag = torch.diagonal(group, dim1=-1, dim2=-2)
    # Find the index of major columns and diagonals
    major = torch.logical_and(
        ddiag[..., 1] > ddiag[..., 0], ddiag[..., 1] > ddiag[..., 2]
    ) + 2 * torch.logical_and(
        ddiag[..., 2] > ddiag[..., 0], ddiag[..., 2] > ddiag[..., 1]
    )
    major = major.view(-1)
    aux = torch.ones(size, dtype=torch.bool)
    sel_rows = 0.5 * (group[aux, major] + group[aux, :, major]).view(*size, 3)
    sel_rows[aux, major] -= cosine.view(-1)
    axis = sel_rows / torch.where(
        near_zero,
        constants._NON_ZERO,
        sel_rows.norm(dim=-1),
    ).view(*size, 1)
    sign_tmp = sine_axis[aux, major].sign().view(size)
    sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
    tangent_vector = torch.where(
        near_pi.view(*size, 1), axis * (theta * sign).view(*size, 1), ret
    )

    return tangent_vector, (theta, sine, cosine)


def _log_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    tangent_vector, _ = _log_impl_helper(group)
    return tangent_vector


def _jlog_impl_helper(
    tangent_vector: torch.Tensor,
    theta: torch.Tensor,
    sine: torch.Tensor,
    cosine: torch.Tensor,
):
    size = get_tangent_vector_size(tangent_vector)
    d_near_zero = theta < LIE_PARAMS.get_eps("so3", "d_near_zero", tangent_vector.dtype)
    theta2 = theta**2
    sine_theta = sine * theta
    two_cosine_minus_two = 2 * cosine - 2
    two_cosine_minus_two_nz = torch.where(
        d_near_zero, constants._NON_ZERO, two_cosine_minus_two
    )
    theta2_nz = torch.where(d_near_zero, constants._NON_ZERO, theta2)

    a = torch.where(d_near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz)
    b = torch.where(
        d_near_zero,
        1.0 / 12 + theta2 / 720,
        (sine_theta + two_cosine_minus_two) / (theta2_nz * two_cosine_minus_two_nz),
    )

    b_tangent_vector = b.view(*size, 1) * tangent_vector
    jac = b_tangent_vector.view(*size, 3, 1) * tangent_vector.view(*size, 1, 3)

    half_ret = 0.5 * tangent_vector
    jac[..., 0, 1] -= half_ret[..., 2]
    jac[..., 1, 0] += half_ret[..., 2]
    jac[..., 0, 2] += half_ret[..., 1]
    jac[..., 2, 0] -= half_ret[..., 1]
    jac[..., 1, 2] -= half_ret[..., 0]
    jac[..., 2, 1] += half_ret[..., 0]

    diag_jac = torch.diagonal(jac, dim1=-1, dim2=-2)
    diag_jac += a.view(*size, 1)

    return jac, (b_tangent_vector,)


def _jlog_impl(group: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group)
    tangent_vector, (theta, sine, cosine) = _log_impl_helper(group)
    jac, _ = _jlog_impl_helper(tangent_vector, theta, sine, cosine)
    return [jac], tangent_vector


def _log_backward(
    group: torch.Tensor, jacobian: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    jacobian = 0.5 * jacobian
    temp = _lift_autograd_fn(
        (jacobian.transpose(-2, -1) @ grad_output.unsqueeze(-1)).squeeze(-1)
    )
    return group @ temp


class Log(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        tangent_vector = _log_impl(group)
        return tangent_vector

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group, ). outputs is tangent_vector
        ctx.save_for_backward(outputs, inputs[0])

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[1]
        return _log_backward(group, _jlog_impl(group)[0][0], grad_output)


class _LogPassthroughWrapper(lie_group._UnaryPassthroughFn):
    @classmethod
    def _backward_impl(
        cls, group: torch.Tensor, jacobian: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        return _log_backward(group, jacobian, grad_output)


# TODO: Implement analytic backward for _jlog_impl
_log_autograd_fn = Log.apply
_jlog_autograd_fn = _jlog_impl
_log_passthrough_fn = _LogPassthroughWrapper.apply


# -----------------------------------------------------------------------------
# Adjoint Transformation
# -----------------------------------------------------------------------------
def _adjoint_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    return group.clone()


# NOTE: No jacobian is defined for the adjoint transformation
_jadjoint_impl = None


class Adjoint(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        return _adjoint_impl(group)

    @classmethod
    def backward(cls, ctx, grad_output):
        return grad_output


_adjoint_autograd_fn = Adjoint.apply
_jadjoint_autograd_fn = None


# -----------------------------------------------------------------------------
# Inverse
# -----------------------------------------------------------------------------
def _inverse_impl(group: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    return group.transpose(-1, -2)


_jinverse_impl = lie_group.JInverseImplFactory(_module)


class Inverse(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, group):
        group: torch.Tensor = cast(torch.Tensor, group)
        return _inverse_impl(group)

    @classmethod
    def backward(cls, ctx, grad_output):
        return grad_output.transpose(-1, -2)


_inverse_autograd_fn = Inverse.apply
_jinverse_autograd_fn = _jinverse_impl


# -----------------------------------------------------------------------------
# Hat
# -----------------------------------------------------------------------------
def _hat_impl(tangent_vector: torch.Tensor) -> torch.Tensor:
    check_tangent_vector(tangent_vector)
    tangent_vector = tangent_vector.squeeze(-1)
    size = get_tangent_vector_size(tangent_vector)
    tensor = tangent_vector.new_zeros(*size, 3, 3)
    tensor[..., 0, 1] = -tangent_vector[..., 2]
    tensor[..., 0, 2] = tangent_vector[..., 1]
    tensor[..., 1, 2] = -tangent_vector[..., 0]
    tensor[..., 1, 0] = tangent_vector[..., 2]
    tensor[..., 2, 0] = -tangent_vector[..., 1]
    tensor[..., 2, 1] = tangent_vector[..., 0]

    return tensor


# NOTE: No jacobian is defined for the hat operator
_jhat_impl = None


class Hat(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _hat_impl(tangent_vector)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return torch.stack(
            (
                grad_output[..., 2, 1] - grad_output[..., 1, 2],
                grad_output[..., 0, 2] - grad_output[..., 2, 0],
                grad_output[..., 1, 0] - grad_output[..., 0, 1],
            ),
            dim=-1,
        )


_hat_autograd_fn = Hat.apply
_jhat_autograd_fn = None


# -----------------------------------------------------------------------------
# Vee
# -----------------------------------------------------------------------------
def _vee_impl(tensor: torch.Tensor) -> torch.Tensor:
    check_hat_tensor(tensor)
    return 0.5 * torch.stack(
        (
            tensor[..., 2, 1] - tensor[..., 1, 2],
            tensor[..., 0, 2] - tensor[..., 2, 0],
            tensor[..., 1, 0] - tensor[..., 0, 1],
        ),
        dim=-1,
    )


# NOTE: No jacobian is defined for the vee operator
_jvee_impl = None


class Vee(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tangent_vector):
        tangent_vector: torch.Tensor = cast(torch.Tensor, tangent_vector)
        ret = _vee_impl(tangent_vector)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return 0.5 * _hat_autograd_fn(grad_output)


_vee_autograd_fn = Vee.apply
_jvee_autograd_fn = None


# -----------------------------------------------------------------------------
# Compose
# -----------------------------------------------------------------------------
def _compose_impl(group0: torch.Tensor, group1: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group0)
    check_group_tensor(group1)
    return group0 @ group1


def _jcompose_impl(
    group0: torch.Tensor, group1: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group0)
    check_group_tensor(group1)
    ret = group0 @ group1
    size = get_group_size(ret)
    jac0 = group1.transpose(-1, -2).expand(*size, 3, 3).clone()
    jac1 = group0.new_zeros(*size, 3, 3)
    jac1[..., _DIAG_3_IDX, _DIAG_3_IDX] = 1
    return [jac0, jac1], ret


class Compose(lie_group.BinaryOperator):
    @classmethod
    def _forward_impl(cls, group0, group1):
        group0: torch.Tensor = cast(torch.Tensor, group0)
        group1: torch.Tensor = cast(torch.Tensor, group1)
        ret = _compose_impl(group0, group1)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group0, group1)
        ctx.save_for_backward(inputs[0], inputs[1])

    @classmethod
    def backward(cls, ctx, grad_output):
        group0, group1 = ctx.saved_tensors
        return (
            grad_output @ group1.transpose(-1, -2),
            group0.transpose(-1, -2) @ grad_output,
        )


_compose_autograd_fn = Compose.apply
_jcompose_autograd_fn = _jcompose_impl


# -----------------------------------------------------------------------------
# Transform
# -----------------------------------------------------------------------------
def _transform_impl(group: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    check_transform_tensor(tensor)
    ret = group @ tensor.unsqueeze(-1)
    return ret.squeeze(-1)


def _jtransform_impl(
    group: torch.Tensor, tensor: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group)
    check_transform_tensor(tensor)
    ret = _transform_impl(group, tensor)
    size = get_transform_tensor_size(ret)
    jacobian_g = -group @ _hat_autograd_fn(tensor)
    jacobian_p = group
    jacobian_g = jacobian_g.expand(*size, 3, 3).clone()
    jacobian_p = jacobian_p.expand(*size, 3, 3).clone()
    return [jacobian_g, jacobian_p], ret


class TransformFrom(lie_group.BinaryOperator):
    @classmethod
    def _forward_impl(cls, group, tensor):
        group: torch.Tensor = cast(torch.Tensor, group)
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _transform_impl(group, tensor)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group, tensor)
        ctx.save_for_backward(inputs[0], inputs[1])

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        tensor: torch.Tensor = ctx.saved_tensors[1]
        grad_output: torch.Tensor = grad_output.unsqueeze(-1)
        tensor_size = get_transform_tensor_size(tensor)
        grad_input0 = grad_output @ tensor.view(*tensor_size, 1, 3)
        grad_input1 = group[..., :3].transpose(-1, -2) @ grad_output
        return grad_input0, grad_input1.squeeze(-1)


_transform_autograd_fn = TransformFrom.apply
_jtransform_autograd_fn = _jtransform_impl


# -----------------------------------------------------------------------------
# Untransform
# -----------------------------------------------------------------------------
def _untransform_impl(group: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    check_group_tensor(group)
    check_transform_tensor(tensor)
    ret = group.transpose(-1, -2) @ tensor.unsqueeze(-1)
    return ret.squeeze(-1)


def _juntransform_impl(
    group: torch.Tensor, tensor: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    check_group_tensor(group)
    check_transform_tensor(tensor)
    ret = _untransform_impl(group, tensor)
    size = get_transform_tensor_size(ret)
    jacobian_g = _hat_autograd_fn(ret)
    jacobian_p = group.transpose(-1, -2)
    jacobian_g = jacobian_g.expand(*size, 3, 3).clone()
    jacobian_p = jacobian_p.expand(*size, 3, 3).clone()
    return [jacobian_g, jacobian_p], ret


class Untransform(lie_group.BinaryOperator):
    @classmethod
    def _forward_impl(cls, group, tensor):
        group: torch.Tensor = cast(torch.Tensor, group)
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _untransform_impl(group, tensor)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group, tensor)
        ctx.save_for_backward(inputs[0], inputs[1])

    @classmethod
    def backward(cls, ctx, grad_output):
        group: torch.Tensor = ctx.saved_tensors[0]
        tensor: torch.Tensor = ctx.saved_tensors[1]
        grad_output: torch.Tensor = grad_output.unsqueeze(-1)
        tensor_size = get_transform_tensor_size(tensor)
        grad_input0 = tensor.view(*tensor_size, 3, 1) @ grad_output.transpose(-1, -2)
        grad_input1 = group[..., :3] @ grad_output
        return grad_input0, grad_input1.squeeze(-1)


_untransform_autograd_fn = Untransform.apply
_juntransform_autograd_fn = _juntransform_impl


# -----------------------------------------------------------------------------
# Unit Quaternion to Rotation Matrix
# -----------------------------------------------------------------------------
def _quaternion_to_rotation_impl(quaternion: torch.Tensor) -> torch.Tensor:
    check_unit_quaternion(quaternion)

    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

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

    size = quaternion.shape[:-1]
    ret = quaternion.new_zeros(*size, 3, 3)
    ret[..., 0, 0] = q00 + q11 - q22 - q33
    ret[..., 0, 1] = 2 * (q12 - q03)
    ret[..., 0, 2] = 2 * (q13 + q02)
    ret[..., 1, 0] = 2 * (q12 + q03)
    ret[..., 1, 1] = q00 - q11 + q22 - q33
    ret[..., 1, 2] = 2 * (q23 - q01)
    ret[..., 2, 0] = 2 * (q13 - q02)
    ret[..., 2, 1] = 2 * (q23 + q01)
    ret[..., 2, 2] = q00 - q11 - q22 + q33
    return ret


def _jquaternion_to_rotation_impl(
    quaternion: torch.Tensor,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    check_unit_quaternion(quaternion)

    quaternion_norm = torch.norm(quaternion, dim=-1, keepdim=True)
    quaternion = quaternion / quaternion_norm
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

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

    size = quaternion.shape[:-1]
    ret = quaternion.new_zeros(*size, 3, 3)
    ret[..., 0, 0] = q00 + q11 - q22 - q33
    ret[..., 0, 1] = 2 * (q12 - q03)
    ret[..., 0, 2] = 2 * (q13 + q02)
    ret[..., 1, 0] = 2 * (q12 + q03)
    ret[..., 1, 1] = q00 - q11 + q22 - q33
    ret[..., 1, 2] = 2 * (q23 - q01)
    ret[..., 2, 0] = 2 * (q13 - q02)
    ret[..., 2, 1] = 2 * (q23 + q01)
    ret[..., 2, 2] = q00 - q11 - q22 + q33

    temp = -2 * quaternion / quaternion_norm
    jac = quaternion.new_zeros(*size, 3, 4)
    jac[..., :, :1] = temp[..., 1:].view(*size, 3, 1)
    jac[..., :, 1:] = _hat_autograd_fn(temp[..., 1:])
    jac[..., 0, 1] = -temp[..., 0]
    jac[..., 1, 2] = -temp[..., 0]
    jac[..., 2, 3] = -temp[..., 0]

    return [jac], ret


class QuaternionToRotation(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, quaternion):
        quaternion: torch.Tensor = cast(torch.Tensor, quaternion)
        ret = _quaternion_to_rotation_impl(quaternion)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (quaternion,). outputs is rotation
        ctx.save_for_backward(inputs[0], outputs)

    @classmethod
    def backward(cls, ctx, grad_output):
        quaternion: torch.Tensor = ctx.saved_tensors[0]
        group: torch.Tensor = ctx.saved_tensors[1]
        jacs = _jquaternion_to_rotation_impl(quaternion)[0][0]
        dR = group.transpose(-1, -2) @ grad_output
        size = quaternion.shape[:-1]
        grad_input = jacs.transpose(-1, -2) @ torch.stack(
            (
                dR[..., 2, 1] - dR[..., 1, 2],
                dR[..., 0, 2] - dR[..., 2, 0],
                dR[..., 1, 0] - dR[..., 0, 1],
            ),
            dim=-1,
        ).view(*size, 3, 1)
        return grad_input.view_as(quaternion)


_quaternion_to_rotation_autograd_fn = QuaternionToRotation.apply
_jquaternion_to_rotation_autograd_fn = _jquaternion_to_rotation_impl


# -----------------------------------------------------------------------------
# Lift
# -----------------------------------------------------------------------------
def _lift_impl(tensor: torch.Tensor) -> torch.Tensor:
    check_lift_tensor(tensor)
    ret = tensor.new_zeros(tensor.shape[:-1] + (3, 3))
    ret[..., 0, 1] = -tensor[..., 2]
    ret[..., 0, 2] = tensor[..., 1]
    ret[..., 1, 2] = -tensor[..., 0]
    ret[..., 1, 0] = tensor[..., 2]
    ret[..., 2, 0] = -tensor[..., 1]
    ret[..., 2, 1] = tensor[..., 0]

    return ret


# NOTE: No jacobian is defined for the project operator
_jlift_impl = None


class Lift(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tensor):
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _lift_impl(tensor)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return _project_autograd_fn(grad_output)


_lift_autograd_fn = Lift.apply
_jlift_autograd_fn = None


# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
def _project_impl(tensor: torch.Tensor) -> torch.Tensor:
    check_project_tensor(tensor)
    return torch.stack(
        (
            tensor[..., 2, 1] - tensor[..., 1, 2],
            tensor[..., 0, 2] - tensor[..., 2, 0],
            tensor[..., 1, 0] - tensor[..., 0, 1],
        ),
        dim=-1,
    )


# NOTE: No jacobian is defined for the project operator
_jproject_impl = None


class Project(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tensor):
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _project_impl(tensor)
        return ret

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output: torch.Tensor = cast(torch.Tensor, grad_output)
        return _lift_autograd_fn(grad_output)


_project_autograd_fn = Project.apply
_jproject_autograd_fn = None


# -----------------------------------------------------------------------------
# Left Act
# -----------------------------------------------------------------------------
def _left_act_impl(
    group: torch.Tensor, tensor: torch.Tensor, dim_out: Optional[int] = None
) -> torch.Tensor:
    check_group_tensor(group)
    check_left_act_tensor(tensor)

    dim_out = tensor.ndim - group.ndim if dim_out is None else dim_out

    if group.ndim + dim_out > tensor.ndim:
        tensor = fill_dims(tensor, group.ndim + dim_out)

    permuted_dim = permute_op_dim(tensor.ndim, dim_out, 2)
    unpermuted_dim = unpermute_op_dim(tensor.ndim, dim_out, 2)
    tensor = tensor.permute(permuted_dim)
    return (group @ tensor).permute(unpermuted_dim)


def _left_act_backward_helper(
    group: torch.Tensor, tensor: torch.Tensor, dim_out: int, grad_output: torch.Tensor
):
    if group.ndim + dim_out > tensor.ndim:
        tensor = fill_dims(tensor, group.ndim + dim_out)

    permuted_dim = permute_op_dim(tensor.ndim, dim_out, 2)
    unpermuted_dim = unpermute_op_dim(tensor.ndim, dim_out, 2)
    tensor = tensor.permute(permuted_dim)
    grad_output = grad_output.permute(permuted_dim)
    jac_group = (grad_output @ tensor.transpose(-1, -2)).permute(unpermuted_dim)
    jac_tensor = (group.transpose(-1, -2) @ grad_output).permute(unpermuted_dim)
    if dim_out > 0:
        dim = list(range(tensor.ndim - 2 - dim_out, tensor.ndim - 2))
        jac_group = jac_group.sum(dim)
    return jac_group, jac_tensor, None


class LeftAct(lie_group.GradientOperator):
    @classmethod
    def _forward_impl(cls, group, tensor, dim_out):
        group: torch.Tensor = cast(torch.Tensor, group)
        tensor: torch.Tensor = cast(torch.Tensor, tensor)
        ret = _left_act_impl(group, tensor, dim_out)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group, tensor)
        ctx.save_for_backward(inputs[0], inputs[1])
        ctx.dim_out = inputs[2]

    @classmethod
    def backward(cls, ctx, grad_output):
        group, tensor = ctx.saved_tensors
        dim_out = ctx.dim_out
        dim_out: int = tensor.ndim - group.ndim if dim_out is None else dim_out
        return _left_act_backward_helper(group, tensor, dim_out, grad_output)


def _left_act_autograd_fn(
    group: torch.Tensor, tensor: torch.Tensor, dim_out: Optional[int] = None
):
    return LeftAct.apply(group, tensor, dim_out)


_jleft_act_autograd_fn = None


# -----------------------------------------------------------------------------
# Left Project
# -----------------------------------------------------------------------------
_left_project_impl = lie_group.LeftProjectImplFactory(_module)
_jleft_project_impl = None


def _left_project_backward_helper(
    group: torch.Tensor,
    tensor: torch.Tensor,
    dim_out: int,
    grad_output_lifted: torch.Tensor,
):
    jac_group, jac_tensor, _ = _left_act_backward_helper(
        group.transpose(-1, -2), tensor, dim_out, grad_output_lifted
    )
    return jac_group.transpose(-1, -2), jac_tensor, None


class LeftProject(lie_group.GradientOperator):
    @classmethod
    def _forward_impl(cls, group, tensor, dim_out):
        group = cast(torch.Tensor, group)
        tensor = cast(torch.Tensor, tensor)
        ret = _left_project_impl(group, tensor, dim_out)
        return ret

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        # inputs is (group, tensor)
        ctx.save_for_backward(inputs[0], inputs[1])
        ctx.dim_out = inputs[2]

    @classmethod
    def backward(cls, ctx, grad_output):
        group, tensor = ctx.saved_tensors
        dim_out = ctx.dim_out
        dim_out: int = tensor.ndim - group.ndim if dim_out is None else dim_out
        grad_output_lifted = _lift_autograd_fn(grad_output)
        return _left_project_backward_helper(group, tensor, dim_out, grad_output_lifted)


def _left_project_autograd_fn(
    group: torch.Tensor, tensor: torch.Tensor, dim_out: Optional[int] = None
):
    return LeftProject.apply(group, tensor, dim_out)


_jleft_project_autograd_fn = _jleft_project_impl


# -----------------------------------------------------------------------------
# Normalize
# -----------------------------------------------------------------------------
def _normalize_impl_helper(tensor: torch.Tensor):
    check_group_shape(tensor)
    size = tensor.shape[:-2]
    u, s, v = torch.svd(tensor)
    sign = torch.det(u @ v).view(*size, 1, 1)
    vt = torch.cat(
        (v[..., :2], torch.where(sign > 0, v[..., 2:], -v[..., 2:])), dim=-1
    ).transpose(-1, -2)
    return u @ vt, {"u": u, "s": s, "v": v, "sign": sign}


def _normalize_impl(tensor: torch.Tensor) -> torch.Tensor:
    return _normalize_impl_helper(tensor)[0]


def _normalize_backward_helper(
    u: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    sign: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    def _skew_symm(tensor: torch.Tensor) -> torch.Tensor:
        return tensor - tensor.transpose(-1, -2)

    size = u.shape[:-2]
    ut = u.transpose(-1, -2)
    vt = v.transpose(-1, -2)
    grad_u: torch.Tensor = grad_output @ torch.cat(
        (v[..., :2], v[..., 2:] @ sign), dim=-1
    )
    grad_v: torch.Tensor = grad_output.transpose(-1, -2) @ torch.cat(
        (u[..., :2], u[..., 2:] @ sign), dim=-1
    )
    s_squared: torch.Tensor = s.pow(2)
    F = s_squared.view(*size, 1, 3).expand(*size, 3, 3) - s_squared.view(
        *size, 3, 1
    ).expand(*size, 3, 3)
    F = torch.where(F == 0, constants._INF, F)
    F = F.pow(-1)

    u_term: torch.Tensor = u @ (F * _skew_symm(ut @ grad_u))
    # The next 3 lines are equivalent to u_term = torch.einsum("n...ij, n...j->n...ij", u_term, s).
    # This implementation is compatible for vectorize=True in torch.autograd.functional.jacobian.
    permuted_u_term_dim = permute_op_dim(u_term.dim(), 1, 1)
    unpermuted_u_term_dim = unpermute_op_dim(u_term.dim(), 1, 1)
    u_term = (u_term.permute(permuted_u_term_dim) * s).permute(unpermuted_u_term_dim)
    u_term = u_term @ vt

    v_term: torch.Tensor = (F * _skew_symm(vt @ grad_v)) @ vt
    # The next 3 lines are equivalent to v_term = torch.einsum("n...i, n...ij->n...ij", s, v_term).
    # This implementation is compatible for vectorize=True in torch.autograd.functional.jacobian.
    permuted_v_term_dim = permute_op_dim(v_term.dim(), 1, 0)
    unpermuted_v_term_dim = unpermute_op_dim(v_term.dim(), 1, 0)
    v_term = (s * v_term.permute(permuted_v_term_dim)).permute(unpermuted_v_term_dim)
    v_term = u @ v_term

    return u_term + v_term


class Normalize(lie_group.UnaryOperator):
    @classmethod
    def _forward_impl(cls, tensor):
        tensor: torch.Tensor = tensor
        output, svd_info = _normalize_impl_helper(tensor)
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
        return _normalize_backward_helper(u, s, v, sign, grad_output), None


def _normalize_autograd_fn(tensor: torch.Tensor):
    return Normalize.apply(tensor)[0]


_jnormalize_autograd_fn = None


_fns = lie_group.LieGroupFns(_module)
