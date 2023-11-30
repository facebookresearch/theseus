# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Any, Callable, List, Optional, Protocol, Tuple

import torch

from torchlie.global_params import _TORCHLIE_GLOBAL_PARAMS as LIE_PARAMS

from .constants import DeviceType
from .utils import check_jacobians_list

# There are four functions associated with each Lie group operator xxx.
# ----------------------------------------------------------------------------------
# _xxx_impl: analytic implementation of the operator, return xxx
# _jxxx_impl: analytic implementation of the operator jacobian, return jxxx and xxx
# _xxx_autograd_fn: a torch.autograd.Function wrapper of _xxx_impl
# _jxxx_autograd_fn: simply equivalent to _jxxx_impl for now
# ----------------------------------------------------------------------------------
# Note that _jxxx_impl might not exist for some operators.
#
# Some operators support a _xxx_passthrough_fn, which returns the same values as
# _xxx_autograd_fn in forward pass, but takes the output of _jxxx_autograd_fn as
# extra non-differentiable inputs to avoid computing operators twice.


def JInverseImplFactory(module):
    def _jinverse_impl(group: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return [-module._adjoint_autograd_fn(group)], module._inverse_autograd_fn(group)

    return _jinverse_impl


def LeftProjectImplFactory(module):
    def _left_project_impl(
        group: torch.Tensor, tensor: torch.Tensor, dim_out: Optional[int] = None
    ) -> torch.Tensor:
        module.check_group_tensor(group)
        module.check_left_project_tensor(tensor)
        group_inverse = module._inverse_autograd_fn(group)

        return module._project_autograd_fn(
            module._left_act_autograd_fn(group_inverse, tensor, dim_out)
        )

    return _left_project_impl


# This class is used by `UnaryOperatorFactory` to
# avoid computing the operator twice in function calls of the form
#      op(group, jacobians_list=jlist).
# This is functionally equivalent to `UnaryOperator` objects, but
# it receives the operator's result and jacobian as extra inputs.
# Usage is then:
#    jac, res = _jop_impl(group)
#    op_result = passthrough_fn(group, res, jac)
# This connects `op_result` to the compute graph with custom
# backward implementation, while `jac` uses torch default autograd.
class _UnaryPassthroughFn(torch.autograd.Function):
    generate_vmap_rule = True

    @classmethod
    @abc.abstractmethod
    def _backward_impl(
        cls, group: torch.Tensor, jacobian: torch.Tensor, grad_output: torch.Tensor
    ) -> torch.Tensor:
        pass

    @classmethod
    def forward(cls, group, op_result, jacobian):
        return op_result

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        ctx.save_for_backward(inputs[0], inputs[2])

    @classmethod
    def backward(cls, ctx, grad_output):
        grad = cls._backward_impl(
            ctx.saved_tensors[0], ctx.saved_tensors[1], grad_output
        )
        return grad, None, None


class UnaryOperator(torch.autograd.Function):
    generate_vmap_rule = True

    @classmethod
    @abc.abstractmethod
    def _forward_impl(cls, tensor: torch.Tensor) -> Any:
        pass

    @classmethod
    def forward(cls, *args):
        assert len(args) in [1, 2]
        if len(args) == 1:  # torch >= 2.0, args is (tensor,)
            output = cls._forward_impl(args[0])
        else:  # args is (ctx, tensor)
            output = cls._forward_impl(args[1])
            cls.setup_context(args[0], (args[1],), output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, outputs):
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
    op_autograd_fn = getattr(module, f"_{op_name}_autograd_fn")
    jop_autograd_fn = getattr(module, f"_j{op_name}_autograd_fn")
    op_passthrough_fn = getattr(module, f"_{op_name}_passthrough_fn", None)

    def op(
        input: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            _check_jacobians_supported(jop_autograd_fn, module.NAME, op_name)
            check_jacobians_list(jacobians)
            jacobians_op, ret = jop_autograd_fn(input)
            jacobians.append(jacobians_op[0])
            if LIE_PARAMS._allow_passthrough_ops and op_passthrough_fn is not None:
                return op_passthrough_fn(input, ret, jacobians_op[0])
        return op_autograd_fn(input)

    def jop(input: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        _check_jacobians_supported(
            jop_autograd_fn, module.name, op_name, is_kwarg=False
        )
        return jop_autograd_fn(input)

    return op, jop


class BinaryOperator(torch.autograd.Function):
    generate_vmap_rule = True

    @classmethod
    @abc.abstractmethod
    def _forward_impl(cls, input0, input1):
        pass

    @classmethod
    def forward(cls, *args):
        assert len(args) in [2, 3]
        if len(args) == 2:  # torch >= 2.0, args is (tensor1, tensor2)
            output = cls._forward_impl(args[0], args[1])
        else:  # args is (ctx, tensor)
            output = cls._forward_impl(args[1], args[2])
            cls.setup_context(args[0], (args[1], args[2]), output)
        return output

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
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


class GradientOperator(torch.autograd.Function):
    generate_vmap_rule = True

    @classmethod
    @abc.abstractmethod
    def _forward_impl(cls, group, tensor, dim_out):
        pass

    @classmethod
    def forward(cls, *args):
        assert len(args) in [3, 4]
        if len(args) == 3:  # torch >= 2.0, args is (group, tensor, dim_out)
            output = cls._forward_impl(args[0], args[1], args[2])
        else:  # args is (ctx, group, tensor, dim_out)
            output = cls._forward_impl(args[1], args[2], args[3])
            cls.setup_context(args[0], (args[1], args[2], args[3]), output)
        return output

    @classmethod
    def setup_context(cls, ctx, inputs, outputs):
        pass


class GradientOperatorOpFnType(Protocol):
    def __call__(
        self,
        group: torch.Tensor,
        tensor: torch.Tensor,
        dim_out: Optional[int] = None,
    ) -> torch.Tensor:
        pass


class GradientOperatorJOpFnType(Protocol):
    def __call__(
        self,
        group: torch.Tensor,
        tensor: torch.Tensor,
        dim_out: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pass


def GradientOperatorFactory(
    module, op_name
) -> Tuple[GradientOperatorOpFnType, GradientOperatorJOpFnType]:
    # Get autograd.Function wrapper of op and its jacobian
    op_autograd_fn = getattr(module, "_" + op_name + "_autograd_fn")
    jop_autograd_fn = getattr(module, "_j" + op_name + "_autograd_fn")

    def op(
        group: torch.Tensor,
        tensor: torch.Tensor,
        dim_out: Optional[int] = None,
    ) -> torch.Tensor:
        return op_autograd_fn(group, tensor, dim_out)

    def jop(
        group: torch.Tensor,
        tensor: torch.Tensor,
        dim_out: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        _check_jacobians_supported(
            jop_autograd_fn, module.NAME, op_name, is_kwarg=False
        )
        return jop_autograd_fn(group, tensor, dim_out)

    return op, jop


# Namespace to facilitate type-checking downstream
class LieGroupFns:
    def __init__(self, module):
        self.exp, self.jexp = UnaryOperatorFactory(module, "exp")
        self.log, self.jlog = UnaryOperatorFactory(module, "log")
        self.adj = UnaryOperatorFactory(module, "adjoint")[0]
        self.normalize = UnaryOperatorFactory(module, "normalize")[0]
        self.inv, self.jinv = UnaryOperatorFactory(module, "inverse")
        self.hat = UnaryOperatorFactory(module, "hat")[0]
        self.vee = UnaryOperatorFactory(module, "vee")[0]
        self.lift = UnaryOperatorFactory(module, "lift")[0]
        self.project = UnaryOperatorFactory(module, "project")[0]
        self.compose, self.jcompose = BinaryOperatorFactory(module, "compose")
        self.left_act = GradientOperatorFactory(module, "left_act")[0]
        self.left_project = GradientOperatorFactory(module, "left_project")[0]
        self.transform, self.jtransform = BinaryOperatorFactory(module, "transform")
        self.untransform, self.juntransform = BinaryOperatorFactory(
            module, "untransform"
        )
        if hasattr(module, "QuaternionToRotation"):
            (
                self.quaternion_to_rotation,
                self.jquaternion_to_rotation,
            ) = UnaryOperatorFactory(module, "quaternion_to_rotation")
        self.check_group_tensor: _CheckFnType = module.check_group_tensor
        self.check_tangent_vector: _CheckFnType = module.check_tangent_vector
        self.check_hat_tensor: _CheckFnType = module.check_hat_tensor
        if hasattr(module, "check_unit_quaternion"):
            self.check_unit_quaternion: _CheckFnType = module.check_unit_quaternion
        self.check_lift_tensor: _CheckFnType = module.check_lift_tensor
        self.check_project_tensor: _CheckFnType = module.check_project_tensor
        self.check_left_act_tensor: _CheckFnType = module.check_left_act_tensor
        self.check_left_project_tensor: _CheckFnType = module.check_left_project_tensor
        self.rand: _RandFnType = module.rand
        self.randn: _RandFnType = module.randn
        self.identity: _IdentityFnType = module.identity
