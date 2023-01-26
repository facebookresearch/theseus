# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import List, Tuple, Union

import torch

from theseus.labs.lie_functional.impl.lie_group import UnaryOperatorOpFnType
from theseus.labs.lie_functional.impl import se3_fns as _se3_base
from theseus.labs.lie_functional.impl import so3_fns as _so3_base


TensorType = Union[torch.Tensor, "LieTensor"]


class LieType(Enum):
    SE3 = 0
    SO3 = 1


SE3 = LieType.SE3
SO3 = LieType.SO3

_JFnReturnType = Tuple[List[torch.Tensor], TensorType]


class LieTensor:
    def __init__(self, data_tensor: torch.Tensor, ltype: LieType):
        self._t = data_tensor
        self._fn_lib = {
            SE3: _se3_base,
            SO3: _so3_base,
        }[ltype]
        self._fn_lib.check_group_tensor(data_tensor)
        self.ltype = ltype

    def __repr__(self) -> str:
        return f"LieTensor({self._t}, ltype=lie.{self.ltype})"

    # Returns a new LieTensor with the given data and the same ltype as self
    def new(self, t: TensorType) -> "LieTensor":
        tensor = t if isinstance(t, torch.Tensor) else t._t
        return LieTensor(tensor.clone(), ltype=self.ltype)

    # Operators
    def exp(self, tangent_vector: torch.Tensor) -> "LieTensor":
        return self.new(self._fn_lib.exp(tangent_vector))

    def log(self, group: "LieTensor") -> torch.Tensor:
        return self._fn_lib.log(group._t)

    def adj(self, group: "LieTensor") -> "LieTensor":
        return self.new(group)

    def inv(self, group: "LieTensor") -> "LieTensor":
        return self.new(self._fn_lib.inv(group._t))

    def hat(self, tangent_vector: torch.Tensor) -> torch.Tensor:
        return self._fn_lib.hat(tangent_vector)

    def vee(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._fn_lib.vee(matrix)

    def lift(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._fn_lib.lift(matrix)

    def project(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._fn_lib.project(matrix)

    # Operator Jacobians
    def _unary_jop_base(
        self,
        input0: torch.Tensor,
        fn: UnaryOperatorOpFnType,
        out_is_group: bool = True,
    ) -> _JFnReturnType:
        jacs: List[torch.Tensor] = []
        op_res: TensorType = fn(input0, jacobians=jacs)
        if out_is_group:
            op_res = self.new(op_res)
        return jacs, op_res

    def jexp(self, tangent_vector: torch.Tensor) -> _JFnReturnType:
        return self._unary_jop_base(tangent_vector, self._fn_lib.exp)

    def jlog(self, group: "LieTensor") -> _JFnReturnType:
        return self._unary_jop_base(group._t, self._fn_lib.exp, out_is_group=False)

    def jinv(self, group: "LieTensor") -> _JFnReturnType:
        return self._unary_jop_base(group._t, self._fn_lib.exp)
