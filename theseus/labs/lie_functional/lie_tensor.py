# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Union

import torch

from theseus.labs.lie_functional.impl import se3_fns as _se3_base
from theseus.labs.lie_functional.impl import so3_fns as _so3_base


TensorType = Union[torch.Tensor, "LieTensor"]


class LieType(Enum):
    SE3 = 0
    SO3 = 1


SE3 = LieType.SE3
SO3 = LieType.SO3


class LieTensor:
    def __init__(self, data_tensor: torch.Tensor, ltype: LieType):
        self._t = data_tensor
        self._base_lib = {
            SE3: _se3_base,
            SO3: _so3_base,
        }[ltype]
        self._base_lib.check_group_tensor(data_tensor)
        self.ltype = ltype

    def __repr__(self) -> str:
        return f"LieTensor({self._t}, ltype=lie.{self.ltype})"

    # Returns a new LieTensor with the given data and the same ltype as self
    def new(self, t: TensorType) -> "LieTensor":
        tensor = t if isinstance(t, torch.Tensor) else t._t
        return LieTensor(tensor.clone(), ltype=self.ltype)

