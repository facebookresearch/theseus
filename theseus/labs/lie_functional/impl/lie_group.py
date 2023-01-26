# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

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
        group_inverse = module.inverse(group)

        return module.project(module.left_act(group_inverse, matrix))

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
            _check_jacobians_supported(jop_autograd_fn, module.name, op_name)
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
            _check_jacobians_supported(jop_autograd_fn, module.name, op_name)
            check_jacobians_list(jacobians)
            jacobians_op = jop_autograd_fn(input0, input1)[0]
            for jacobian in jacobians_op:
                jacobians.append(jacobian)
        return op_autograd_fn(input0, input1)

    def jop(
        input0: torch.Tensor, input1: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        _check_jacobians_supported(
            jop_autograd_fn, module.name, op_name, is_kwarg=False
        )
        return jop_autograd_fn(input0, input1)

    return op, jop
