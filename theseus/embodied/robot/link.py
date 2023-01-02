# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Optional
import torch

from theseus.geometry.functional import se3


class Link(abc.ABC):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: int = -1,
        child: int = -1,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if origin is None and dtype is None:
            dtype = torch.get_default_dtype()
        if origin is not None:
            if origin.shape[0] != 1 or not se3.check_group_tensor(origin):
                raise ValueError("Origin must be an element of SE(3).")

            if dtype is not None and origin.dtype != dtype:
                warnings.warn(
                    f"tensor.dtype {origin.dtype} does not match given dtype {dtype}, "
                    "tensor.dtype will take precendence."
                )
            dtype = origin.dtype
        else:
            origin = torch.zeros(1, 3, 4, dtype=dtype)
            origin[:, 0, 0] = 1
            origin[:, 1, 1] = 1
            origin[:, 2, 2] = 1

        self._name = name
        self._id = id
        self._parent = parent
        self._child = child
        self._origin = origin
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    @property
    def parent(self) -> int:
        return self._parent

    @property
    def child(self) -> int:
        return self._child

    @property
    def origin(self) -> torch.Tensor:
        return self._origin

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_parent(self, parent: int):
        self._parent = parent

    def set_child(self, child: int):
        self._child = child
