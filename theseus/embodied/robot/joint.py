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
        self._parent = parent
        self._child = child
        self._origin = origin
        self._dtype = dtype
        self._axis: torch.Tensor = torch.zeros(6, 1, dtype=dtype)

    @property
    def name(self) -> str:
        return self._name

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

    @property
    def axis(self) -> torch.Tensor:
        return self._axis

    def set_parent(self, parent: int):
        self._parent = parent

    def set_child(self, child: int):
        self._child = child

    @abc.abstractmethod
    def dof(self) -> int:
        pass

    @abc.abstractmethod
    def relative_pose(self, angle: torch.Tensor) -> torch.Tensor:
        pass


class _RevoluteJointImpl(Joint):
    def __init__(
        self,
        name: str,
        parent: int = -1,
        child: int = -1,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name, parent, child, origin, dtype)

    def dof(self) -> int:
        return 1

    @abc.abstractmethod
    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    def relative_pose(self, angle: torch.Tensor) -> torch.Tensor:
        rot = self._rotation_impl(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3] @ rot
        ret[:, :, 3] = self.origin[:, :, 3]
        return ret


class RevoluteJoint(_RevoluteJointImpl):
    def __init__(
        self,
        angle_axis: torch.Tensor,
        name: str,
        parent: int = -1,
        child: int = -1,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if angle_axis.ndim == 1:
            angle_axis = angle_axis.view(-1, 1)

        if angle_axis.ndim != 2 or angle_axis.shape != (3, 1):
            raise ValueError("The angle axis must be a 3-D vector.")

        super().__init__(name, parent, child, origin, dtype)

        if angle_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of angle_axis should be {self.dtype}.")

        self._axis[3:] = angle_axis

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        return so3.exp(angle @ self.axis[3:].view(1, -1))


class RevoluteJointX(_RevoluteJointImpl):
    def __init__(
        self,
        name: str,
        parent: int = -1,
        child: int = -1,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name, parent, child, origin, dtype)
        self._axis[3] = 1

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 2 and angle.shape[1] == 1:
            angle = angle.view(-1)
        if angle.ndim != 1:
            raise ValueError("The joint angle must be a vector.")
        rot = angle.new_zeros(angle.shape[0], 3, 3)
        rot[:, 0, 0] = 1
        rot[:, 1, 1] = angle.cos()
        rot[:, 2, 1] = angle.sin()
        rot[:, 1, 2] = -rot[:, 2, 1]
        rot[:, 2, 2] = rot[:, 1, 1]
        return rot


class RevoluteJointY(_RevoluteJointImpl):
    def __init__(
        self,
        name: str,
        parent: int = -1,
        child: int = -1,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name, parent, child, origin, dtype)
        self._axis[4] = 1

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 2 and angle.shape[1] == 1:
            angle = angle.view(-1)
        if angle.ndim != 1:
            raise ValueError("The joint angle must be a vector.")
        rot = angle.new_zeros(angle.shape[0], 3, 3)
        rot[:, 1, 1] = 1
        rot[:, 0, 0] = angle.cos()
        rot[:, 0, 2] = angle.sin()
        rot[:, 2, 0] = -rot[:, 0, 2]
        rot[:, 2, 2] = rot[:, 0, 0]
        return rot


class RevoluteJointZ(_RevoluteJointImpl):
    def __init__(
        self,
        name: str,
        parent: int = -1,
        child: int = -1,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name, parent, child, origin, dtype)
        self._axis[5] = 1

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 2 and angle.shape[1] == 1:
            angle = angle.view(-1)
        if angle.ndim != 1:
            raise ValueError("The joint angle must be a vector.")
        rot = angle.new_zeros(angle.shape[0], 3, 3)
        rot[:, 2, 2] = 1
        rot[:, 0, 0] = angle.cos()
        rot[:, 1, 0] = angle.sin()
        rot[:, 0, 1] = -rot[:, 1, 0]
        rot[:, 1, 1] = rot[:, 0, 0]
        return rot
