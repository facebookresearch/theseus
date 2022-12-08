# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import count
from typing import Any, List, Optional, Sequence, Union

import torch
from theseus.constants import DeviceType


class Variable:
    """A variable in a differentiable optimization problem."""

    _ids = count(0)

    def __init__(self, tensor: torch.Tensor, name: Optional[str] = None):
        self._id = next(Variable._ids)
        self._num_updates = 0
        if name:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}__{self._id}"
        self.tensor = tensor

    def copy(self, new_name: Optional[str] = None) -> "Variable":
        if not new_name:
            new_name = f"{self.name}_copy"
        return Variable(self.tensor.clone(), name=new_name)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    # batch_ignore_mask is a boolean list where batch_ignore_mask[i] = 1 means
    # variable[i] will *not* be updated
    def update(
        self,
        data: Union[torch.Tensor, "Variable"],
        batch_ignore_mask: Optional[torch.Tensor] = None,
    ):
        if isinstance(data, Variable):
            tensor = data.tensor
        else:
            tensor = data
        if (
            len(tensor.shape) != len(self.tensor.shape)
            or tensor.shape[1:] != self.tensor.shape[1:]
        ):
            raise ValueError(
                f"Tried to update tensor {self.name} with data "
                f"incompatible with original tensor shape. Given {tensor.shape[1:]}. "
                f"Expected: {self.tensor.shape[1:]}"
            )
        if tensor.dtype != self.dtype:
            raise ValueError(
                f"Tried to update used tensor of dtype {tensor.dtype} but Variable "
                f"{self.name} has dtype {self.dtype}."
            )
        if batch_ignore_mask is not None and batch_ignore_mask.any():
            mask_shape = (-1,) + (1,) * (tensor.ndim - 1)
            self.tensor = torch.where(
                batch_ignore_mask.view(mask_shape), self.tensor, tensor
            )
        else:
            self.tensor = tensor
        self._num_updates += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tensor={self.tensor}, name={self.name})"

    def __str__(self) -> str:
        return repr(self)

    # calls to() on the internal tensors
    def to(self, *args, **kwargs):
        self.tensor = self.tensor.to(*args, **kwargs)

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    def __getitem__(self, item):
        return self.tensor[item]

    def __setitem__(self, item, value):
        self.tensor[item] = value


# If value is a variable, this returns the same variable
# Otherwise value is wrapper into a variable (and a tensor, if needed)
# In this case, the device, dtype and name can be specified.
def as_variable(
    value: Union[float, Sequence[float], torch.Tensor, Variable],
    device: DeviceType = None,
    dtype: Optional[torch.dtype] = None,
    name: Optional[str] = None,
) -> Variable:
    if isinstance(value, Variable):
        return value
    if isinstance(device, str):
        # mypy doesn't like passing str to torch.as_tensor so convert
        device = torch.device(device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if isinstance(value, float):
        tensor = tensor.view(1, 1)
    return Variable(tensor, name=name)


# This context can be used to temporarily use a masked version of a list of
# variables. For example,
#
# with masked_variables([v1, v2], mask):
#   some_computation(v1, v2)
#
# Performs `some_computation()` using v1.tensor[mask] and v2.tensor[mask].
class masked_variables:
    def __init__(self, vars: List[Variable], mask: torch.Tensor) -> None:
        assert mask.dtype == torch.bool and mask.ndim == 1
        self._vars = vars
        self._original_tensors = [v.tensor for v in vars]
        self._mask = mask

    def __enter__(self) -> None:
        for v in self._vars:
            assert v.shape[0] == self._mask.shape[0]
            v.update(v.tensor[self._mask])

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for i, v in enumerate(self._vars):
            v.update(self._original_tensors[i])
