# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import count
from typing import Optional, Union

import torch


class Variable:
    _ids = count(0)

    def __init__(self, data: torch.Tensor, name: Optional[str] = None):
        self._id = next(Variable._ids)
        if name:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}__{self._id}"
        self.data = data

    def copy(self, new_name: Optional[str] = None) -> "Variable":
        if not new_name:
            new_name = f"{self.name}_copy"
        return Variable(self.data.clone(), name=new_name)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    # batch_ignore_mask is a boolean list where batch_ignore_mask[i] = 1 means
    # variable[i] will *not* be updated
    # keep_data is a boolean indicating whether to reuse self.data
    def update(
        self,
        data: Union[torch.Tensor, "Variable"],
        batch_ignore_mask: Optional[torch.Tensor] = None,
        keep_data: Optional[bool] = None,
    ):
        if isinstance(data, Variable):
            data = data.data
        if (
            len(data.shape) != len(self.data.shape)
            or data.shape[1:] != self.data.shape[1:]
        ):
            raise ValueError(
                f"Tried to update tensor {self.name} with data "
                f"incompatible with original tensor shape. Given {data.shape[1:]}. "
                f"Expected: {self.data.shape[1:]}"
            )
        if data.dtype != self.dtype:
            raise ValueError(
                f"Tried to update used tensor of dtype {data.dtype} but Variable "
                f"{self.name} has dtype {self.dtype}."
            )
        if batch_ignore_mask is not None and batch_ignore_mask.any():
            mask_shape = (-1,) + (1,) * (data.ndim - 1)
            if keep_data:
                self.data[:] = torch.where(
                    batch_ignore_mask.view(mask_shape), self.data, data
                )
            else:
                self.data = torch.where(
                    batch_ignore_mask.view(mask_shape), self.data, data
                )
        else:
            if keep_data:
                self.data[:] = data
            else:
                self.data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        return repr(self)

    # calls to() on the internal tensors
    def to(self, *args, **kwargs):
        self.data = self.data.to(*args, **kwargs)

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, value):
        self.data[item] = value
