# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union, TYPE_CHECKING

import torch
from theseus.labs.lie.functional import SE3 as _se3_impl, SO3 as _so3_impl
from theseus.labs.lie.functional.constants import DeviceType
from theseus.labs.lie.functional.lie_group import LieGroupFns

if TYPE_CHECKING:
    from .lie_tensor import LieTensor, _LieTensorBase


# The next two are similar to the ones in functional, except they return a LieTensor.
class _RandFnType(Protocol):
    def __call__(
        self,
        *size: Any,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "LieTensor":
        pass


class _IdentityFnType(Protocol):
    def __call__(
        self,
        *size: int,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "LieTensor":
        pass


Device = Union[torch.device, str, None]
TensorType = Union[torch.Tensor, "_LieTensorBase"]
_JFnReturnType = Tuple[List[torch.Tensor], TensorType]


class _ltype(Enum):
    SE3 = 0
    SO3 = 1


def _get_fn_lib(ltype: _ltype):
    return {
        _ltype.SE3: _se3_impl,
        _ltype.SO3: _so3_impl,
    }[ltype]


def _eval_op(
    fn_lib: LieGroupFns,
    op_name: str,
    input0: torch.Tensor,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return getattr(fn_lib, op_name)(input0, jacobians=jacobians)


class ltype:
    def __init__(self, lt: _ltype):
        self._lt = lt
        self._fn_lib = _get_fn_lib(lt)

    rand: _RandFnType
    randn: _RandFnType
    identity: _IdentityFnType
    _call_impl: Callable[[torch.Tensor], "LieTensor"]

    def __call__(self, tensor: torch.Tensor) -> "LieTensor":
        return self._call_impl(tensor)

    _create_lie_tensor: Callable[[torch.Tensor, "ltype"], "LieTensor"]

    def exp(self, tangent_vector: torch.Tensor) -> "LieTensor":
        return self._create_lie_tensor(
            _eval_op(self._fn_lib, "exp", tangent_vector), self
        )

    def jexp(self, tangent_vector: torch.Tensor) -> _JFnReturnType:
        jacs: List[torch.Tensor] = []
        op_res: TensorType = self._fn_lib.exp(tangent_vector, jacobians=jacs)
        return jacs, self._create_lie_tensor(op_res, self)

    def hat(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "hat", tangent_vector)

    def vee(self, matrix: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "vee", matrix)

    def lift(self, matrix: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "lift", matrix)

    def project(self, matrix: torch.Tensor) -> torch.Tensor:
        return _eval_op(self._fn_lib, "project", matrix)

    def __str__(self) -> str:
        return self._lt.name


SE3 = ltype(_ltype.SE3)
SO3 = ltype(_ltype.SO3)
