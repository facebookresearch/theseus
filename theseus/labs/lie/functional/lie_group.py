# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from .constants import DeviceType
from typing import Callable, List, Tuple, Optional, Protocol
from .utils import check_jacobians_list

# There are four functions associated with each Lie group operator xxx.
# ----------------------------------------------------------------------------------
# _xxx_impl: analytic implementation of the operator, return xxx
# _jxxx_impl: analytic implementation of the operator jacobian, return jxxx and xxx
# _xxx_autograd_fn: a torch.autograd.Function wrapper of _xxx_impl
# _jxxx_autograd_fn: simply equivalent to _jxxx_impl for now
# ----------------------------------------------------------------------------------
# Note that _jxxx_impl might not exist for some operators.


def JInverseImplFactory(module):
    def _jinverse_impl(group: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return [-module._adjoint_autograd_fn(group)], module._inverse_autograd_fn(group)

    return _jinverse_impl


def LeftProjectImplFactory(module):
    def _left_project_impl(group: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        module.check_group_tensor(group)
        module.check_left_project_matrix(matrix)
        group_inverse = module._inverse_autograd_fn(group)

        return module._project_autograd_fn(
            module._left_act_autograd_fn(group_inverse, matrix)
        )

    return _left_project_impl


class UnaryOperator(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, input):
        pass


class UnaryOperatorOpFnType(Protocol):
    def __call__(
        self, input: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        pass


class UnaryOperatorJOpFnType(Protocol):
    def __call__(self, input: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass


def _check_jacobians_supported(
    jop_autograd_fn: Optional[Callable],
    module_name: str,
    op_name: str,
    is_kwarg: bool = True,
):
    if jop_autograd_fn is None:
        if is_kwarg:
            msg = f"Passing jacobians= is not supported by {module_name}.{op_name}"
        else:
            msg = f"{module_name}.j{op_name} is not implemented."
        raise NotImplementedError(msg)


def UnaryOperatorFactory(
    module, op_name
) -> Tuple[UnaryOperatorOpFnType, UnaryOperatorJOpFnType]:
    # Get autograd.Function wrapper of op and its jacobian
    op_autograd_fn = getattr(module, "_" + op_name + "_autograd_fn")
    jop_autograd_fn = getattr(module, "_j" + op_name + "_autograd_fn")

    def op(
        input: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            _check_jacobians_supported(jop_autograd_fn, module.NAME, op_name)
            check_jacobians_list(jacobians)
            jacobians_op = jop_autograd_fn(input)[0]
            jacobians.append(jacobians_op[0])
        return op_autograd_fn(input)

    def jop(input: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        _check_jacobians_supported(
            jop_autograd_fn, module.name, op_name, is_kwarg=False
        )
        return jop_autograd_fn(input)

    return op, jop


class BinaryOperator(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, input0, input1):
        pass


class BinaryOperatorOpFnType(Protocol):
    def __call__(
        self,
        input0: torch.Tensor,
        input1: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        pass


class BinaryOperatorJOpFnType(Protocol):
    def __call__(
        self, input0: torch.Tensor, input1: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass


def BinaryOperatorFactory(
    module, op_name
) -> Tuple[BinaryOperatorOpFnType, BinaryOperatorJOpFnType]:
    # Get autograd.Function wrapper of op and its jacobian
    op_autograd_fn = getattr(module, "_" + op_name + "_autograd_fn")
    jop_autograd_fn = getattr(module, "_j" + op_name + "_autograd_fn")

    def op(
        input0: torch.Tensor,
        input1: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            _check_jacobians_supported(jop_autograd_fn, module.NAME, op_name)
            check_jacobians_list(jacobians)
            jacobians_op = jop_autograd_fn(input0, input1)[0]
            for jacobian in jacobians_op:
                jacobians.append(jacobian)
        return op_autograd_fn(input0, input1)

    def jop(
        input0: torch.Tensor, input1: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        _check_jacobians_supported(
            jop_autograd_fn, module.NAME, op_name, is_kwarg=False
        )
        return jop_autograd_fn(input0, input1)

    return op, jop


_CheckFnType = Callable[[torch.Tensor], None]


class _RandFnType(Protocol):
    def __call__(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass


class _IdentityFnType(Protocol):
    def __call__(
        *size: int,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass


# Namespace to facilitate type-checking downstream
class LieGroupFns:
    def __init__(self, module):
        self.exp, self.jexp = UnaryOperatorFactory(module, "exp")
        self.log, self.jlog = UnaryOperatorFactory(module, "log")
        self.adj = UnaryOperatorFactory(module, "adjoint")[0]
        self.inv, self.jinv = UnaryOperatorFactory(module, "inverse")
        self.hat = UnaryOperatorFactory(module, "hat")[0]
        self.vee = UnaryOperatorFactory(module, "vee")[0]
        self.lift = UnaryOperatorFactory(module, "lift")[0]
        self.project = UnaryOperatorFactory(module, "project")[0]
        self.compose, self.jcompose = BinaryOperatorFactory(module, "compose")
        self.left_act = BinaryOperatorFactory(module, "left_act")[0]
        self.left_project = BinaryOperatorFactory(module, "left_project")[0]
        self.transform_from, self.jtransform_from = BinaryOperatorFactory(
            module, "transform_from"
        )
        self.check_group_tensor: _CheckFnType = module.check_group_tensor
        self.check_tangent_vector: _CheckFnType = module.check_tangent_vector
        self.check_hat_matrix: _CheckFnType = module.check_hat_matrix
        if hasattr(module, "check_unit_quaternion"):
            self.check_unit_quaternion: _CheckFnType = module.check_unit_quaternion
        if hasattr(module, "check_lift_matrix"):
            self.check_lift_matrix: _CheckFnType = module.check_lift_matrix
        if hasattr(module, "check_project_matrix"):
            self.check_project_matrix: _CheckFnType = module.check_project_matrix
        self.check_left_act_matrix: _CheckFnType = module.check_left_act_matrix
        self.check_left_project_matrix: _CheckFnType = module.check_left_project_matrix
        self.rand: _RandFnType = module.rand
        self.randn: _RandFnType = module.randn
        self.identity: _IdentityFnType = module.identity
