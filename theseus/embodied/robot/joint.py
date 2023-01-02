# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Optional
import torch

from theseus.geometry.functional import so3, se3


class Joint(abc.ABC):
    def __init__(
        self,
        name: str,
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

        self._origin = origin
        self._dtype = dtype
        self._name = name
        self._axis: torch.Tensor = torch.zeros(6, 1, dtype=dtype)

    @property
    def origin(self) -> torch.Tensor:
        return self._origin

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def axis(self) -> torch.Tensor:
        return self._axis

    @abc.abstractmethod
    def dof(self) -> int:
        pass

    @abc.abstractmethod
    def relative_pose(self, angle: torch.Tensor) -> torch.Tensor:
        pass


class RevoluteJoint(Joint):
    def __init__(
        self,
        name: str,
        angle_axis: torch.Tensor,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if angle_axis.ndim == 1:
            angle_axis = angle_axis.view(-1, 1)

        if angle_axis.ndim != 2 or angle_axis.shape != (3, 1):
            raise ValueError("The angle axis must be a 3-D vector.")

        super().__init__(name, origin, dtype)

        if angle_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of angle_axis should be {self.dtype}.")

        self._axis[3:] = angle_axis

    def dof(self) -> int:
        return 1

    def relative_pose(self, angle: torch.Tensor) -> torch.Tensor:
        rot = self.origin[:, :, :3] @ so3.exp(
            angle.view(-1, 1) @ self.axis[3:].view(1, -1)
        )
        t = self.origin[:, :, 3:].expand(angle.shape[:1] + (3, 1))
        return torch.cat((rot, t), dim=-1)
