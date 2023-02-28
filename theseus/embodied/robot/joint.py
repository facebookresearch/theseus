# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Optional
import torch

from .link import Link
from theseus.labs.lie_functional import so3, se3
from theseus.constants import DeviceType


class Joint(abc.ABC):
    def __init__(
        self,
        name: str,
        dof: int,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if origin is None and dtype is None:
            dtype = torch.get_default_dtype()
        if origin is None and device is None:
            device = torch.device("cpu")
        if origin is not None:
            self.set_origin(origin)

            if dtype is not None and origin.dtype != dtype:
                warnings.warn(
                    f"tensor.dtype {origin.dtype} does not match given dtype {dtype}, "
                    "tensor.dtype will take precendence."
                )
            dtype = origin.dtype

            if device is not None and origin.device != device:
                warnings.warn(
                    f"tensor.device {origin.device} does not match given device {device}, "
                    "tensor.device will take precendence."
                )
            dtype = origin.dtype
            device = origin.device
        else:
            origin = torch.zeros(1, 3, 4, dtype=dtype, device=device)
            origin[:, 0, 0] = 1
            origin[:, 1, 1] = 1
            origin[:, 2, 2] = 1

        self._name = name
        self._dof = dof
        self._id = id
        self._parent = parent
        self._child = child
        self._origin = origin
        self._dtype = dtype
        self._device = torch.device(device)
        self._axis: torch.Tensor = torch.zeros(6, self.dof, dtype=dtype, device=device)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def id(self) -> int:
        return self._id

    @property
    def parent(self) -> Optional[Link]:
        return self._parent

    @property
    def child(self) -> Optional[Link]:
        return self._child

    @property
    def origin(self) -> torch.Tensor:
        return self._origin

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def axis(self) -> torch.Tensor:
        return self._axis

    def set_id(self, id: int):
        self._id = id

    def set_parent(self, parent: Optional[Link]):
        self._parent = parent

    def set_child(self, child: Optional[Link]):
        self._child = child

    def set_origin(self, origin: torch.Tensor):
        if origin.shape[0] != 1 or not se3.check_group_tensor(origin):
            raise ValueError("Origin must be an element of SE(3).")
        self._origin = origin

    @abc.abstractmethod
    def relative_pose(self, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class FixedJoint(Joint):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 0, id, parent, child, origin, dtype, device)

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 0:
            raise ValueError("Fixed joint has no inputs.")
        return self.origin


class _RevoluteJointImpl(Joint):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent, child, origin, dtype, device)

    @abc.abstractmethod
    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Revolute joint requires one input.")
        angle: torch.Tensor = args[0]
        rot = self._rotation_impl(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3] @ rot
        ret[:, :, 3] = self.origin[:, :, 3]
        return ret


class RevoluteJoint(_RevoluteJointImpl):
    def __init__(
        self,
        name: str,
        revolute_axis: torch.Tensor,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if revolute_axis.ndim == 1:
            revolute_axis = revolute_axis.view(-1, 1)

        if revolute_axis.ndim != 2 or revolute_axis.shape != (3, 1):
            raise ValueError("The revolute axis must be a 3-D vector.")

        super().__init__(name, id, parent, child, origin, dtype, device)

        if revolute_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of revolute_axis should be {self.dtype}.")

        if revolute_axis.device != self.device:
            raise ValueError(f"The device of revolute_axis should be {self.device}.")

        self._axis[3:] = revolute_axis

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
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent, child, origin, dtype, device)
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
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent, child, origin, dtype, device)
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
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent, child, origin, dtype, device)
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


class _PrismaticJointImpl(Joint):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent, child, origin, dtype, device)

    @abc.abstractmethod
    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Prismatic joint requires one input.")
        angle: torch.Tensor = args[0]
        trans = self._translation_impl(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3:] = (
            self.origin[:, :, :3] @ trans.view(-1, 3, 1) + self.origin[:, :, 3:]
        )
        return ret


class PrismaticJoint(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        prismatic_axis: torch.Tensor,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if prismatic_axis.ndim == 1:
            prismatic_axis = prismatic_axis.view(-1, 1)

        if prismatic_axis.ndim != 2 or prismatic_axis.shape != (3, 1):
            raise ValueError("The prismatic axis must be a 3-D vector.")

        super().__init__(name, id, parent, child, origin, dtype, device)

        if prismatic_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of prismatic_axis should be {self.dtype}.")

        if prismatic_axis.device != self.device:
            raise ValueError(f"The device of prismatic_axis should be {self.device}.")

        self._axis[:3] = prismatic_axis

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        return angle @ self.axis[:3].view(1, -1)


class PrismaticJointX(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent, child, origin, dtype, device)
        self._axis[3] = 1

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 2 and angle.shape[1] == 1:
            angle = angle.view(-1)
        if angle.ndim != 1:
            raise ValueError("The joint angle must be a vector.")
        trans = angle.new_zeros(angle.shape[0], 3)
        trans[:, 0] = angle
        return trans

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Prismatic joint requires one input.")
        angle: torch.Tensor = args[0]
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3] = angle * self.origin[:, :, 0] + self.origin[:, :, 3]
        return ret


class PrismaticJointY(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent, child, origin, dtype, device)
        self._axis[4] = 1

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 2 and angle.shape[1] == 1:
            angle = angle.view(-1)
        if angle.ndim != 1:
            raise ValueError("The joint angle must be a vector.")
        trans = angle.new_zeros(angle.shape[0], 3)
        trans[:, 1] = angle
        return trans

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Prismatic joint requires one input.")
        angle: torch.Tensor = args[0]
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3] = angle * self.origin[:, :, 1] + self.origin[:, :, 3]
        return ret


class PrismaticJointZ(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent: Optional[Link] = None,
        child: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent, child, origin, dtype, device)
        self._axis[5] = 1

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        if angle.ndim == 2 and angle.shape[1] == 1:
            angle = angle.view(-1)
        if angle.ndim != 1:
            raise ValueError("The joint angle must be a vector.")
        trans = angle.new_zeros(angle.shape[0], 3)
        trans[:, 2] = angle
        return trans

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Prismatic joint requires one input.")
        angle: torch.Tensor = args[0]
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3] = angle * self.origin[:, :, 2] + self.origin[:, :, 3]
        return ret
