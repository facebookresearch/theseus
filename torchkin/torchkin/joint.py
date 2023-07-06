# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import List, Optional

import torch

from torchlie.functional import SO3
from torchlie.functional.constants import DeviceType


class Link(abc.ABC):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_joint: "Joint" = None,
        child_joints: Optional[List["Joint"]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._name = name
        self._id = id
        # self._parent_joint is None means that the link is rigidly connected to the world
        self._parent_joint = parent_joint
        self._child_joints = child_joints if child_joints else []
        self._ancestor_links: List[Link] = []
        self._ancestor_non_fixed_joint_ids: List[int] = []
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    @property
    def parent_link(self) -> "Link":
        return (
            self._parent_joint.parent_link if self._parent_joint is not None else None
        )

    @property
    def parent_joint(self) -> "Joint":
        return self._parent_joint

    @property
    def child_joints(self) -> List["Joint"]:
        return self._child_joints

    @property
    def ancestor_links(self) -> List["Link"]:
        return self._ancestor_links

    @property
    def ancestor_non_fixed_joint_ids(self):
        return self._ancestor_non_fixed_joint_ids

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _update_ancestor_links(self):
        curr = self.parent_link
        self._ancestor_links = []
        while curr is not None:
            self._ancestor_links.insert(0, curr)
            curr = curr.parent_link

    def _add_child_joint(self, child_joint: "Joint"):
        self._child_joints.append(child_joint)

    def _remove_child_joint(self, child_joint: "Joint"):
        self._child_joints.remove(child_joint)


class Joint(abc.ABC):
    def __init__(
        self,
        name: str,
        dof: int,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if origin is None and dtype is None:
            dtype = torch.get_default_dtype()
        if origin is None and device is None:
            device = torch.device("cpu")
        if origin is not None:
            self._origin = origin

            if dtype is not None and origin.dtype != dtype:
                warnings.warn(
                    f"The origin's dtype {origin.dtype} does not match given dtype {dtype}, "
                    "Origin's dtype will take precendence."
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
        self._parent_link = parent_link
        self._child_link = child_link
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
    def parent_link(self) -> Optional[Link]:
        return self._parent_link

    @property
    def child_link(self) -> Optional[Link]:
        return self._child_link

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

    @abc.abstractmethod
    def relative_pose(self, angle: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class FixedJoint(Joint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 0, id, parent_link, child_link, origin, dtype, device)

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 0:
            raise ValueError("Fixed joint has no inputs.")
        return self.origin


class _RevoluteJointImpl(Joint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent_link, child_link, origin, dtype, device)

    @abc.abstractmethod
    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _check_input(angle: torch.Tensor):
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")

    def rotation(self, angle: torch.Tensor) -> torch.Tensor:
        _RevoluteJointImpl._check_input(angle)
        return self._rotation_impl(angle.view(-1, 1))

    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        rot = self.rotation(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3] @ rot
        ret[:, :, 3] = self.origin[:, :, 3]
        return ret

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Revolute joint requires one input.")
        angle: torch.Tensor = args[0]
        return self._relative_pose_impl(angle)


class RevoluteJoint(_RevoluteJointImpl):
    def __init__(
        self,
        name: str,
        revolute_axis: torch.Tensor,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if revolute_axis.numel() != 3:
            raise ValueError("The revolute axis must have 3 elements.")

        super().__init__(name, id, parent_link, child_link, origin, dtype, device)

        if revolute_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of revolute_axis should be {self.dtype}.")

        if revolute_axis.device != self.device:
            raise ValueError(f"The device of revolute_axis should be {self.device}.")

        self._axis[3:] = revolute_axis.view(-1, 1)

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        return SO3.exp(angle @ self.axis[3:].view(1, -1))


class _RevoluteJointXYZImpl(_RevoluteJointImpl):
    axis_info = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    def __init__(
        self,
        name: str,
        axis_id: int,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent_link, child_link, origin, dtype, device)
        if axis_id < 0 or axis_id >= 3:
            raise ValueError("The axis_id must be one of (0, 1, 2).")
        self._axis_id = axis_id
        self._axis[self.axis_id + 3] = 1

    @property
    def axis_id(self) -> int:
        return self._axis_id

    def _rotation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        info = _RevoluteJointXYZImpl.axis_info[self.axis_id]
        rot = angle.new_zeros(angle.shape[0], 3, 3)
        rot[:, info[0], info[0]] = 1
        rot[:, info[1], info[1]] = angle.cos()
        rot[:, info[2], info[2]] = rot[:, info[1], info[1]]
        rot[:, info[2], info[1]] = angle.sin()
        rot[:, info[1], info[2]] = -rot[:, info[2], info[1]]

        return rot


class RevoluteJointX(_RevoluteJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, 0, parent_link, child_link, origin, dtype, device)


class RevoluteJointY(_RevoluteJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, 1, parent_link, child_link, origin, dtype, device)


class RevoluteJointZ(_RevoluteJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, 2, parent_link, child_link, origin, dtype, device)


class _PrismaticJointImpl(Joint):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, 1, id, parent_link, child_link, origin, dtype, device)

    @abc.abstractmethod
    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def _check_input(angle: torch.Tensor):
        if angle.ndim == 1:
            angle = angle.view(-1, 1)
        if angle.ndim != 2 or angle.shape[1] != 1:
            raise ValueError("The joint angle must be a vector.")

    def translation(self, angle: torch.Tensor) -> torch.Tensor:
        _PrismaticJointImpl._check_input(angle)
        return self._translation_impl(angle.view(-1, 1))

    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        trans = self.translation(angle)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3:] = (
            self.origin[:, :, :3] @ trans.view(-1, 3, 1) + self.origin[:, :, 3:]
        )
        return ret

    def relative_pose(self, *args) -> torch.Tensor:
        if len(args) != 1:
            raise ValueError("Prismatic joint requires one input.")
        angle: torch.Tensor = args[0]
        return self._relative_pose_impl(angle)


class PrismaticJoint(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        prismatic_axis: torch.Tensor,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        if prismatic_axis.numel() != 3:
            raise ValueError("The prismatic axis must have 3 elements.")

        super().__init__(name, id, parent_link, child_link, origin, dtype, device)

        if prismatic_axis.dtype != self.dtype:
            raise ValueError(f"The dtype of prismatic_axis should be {self.dtype}.")

        if prismatic_axis.device != self.device:
            raise ValueError(f"The device of prismatic_axis should be {self.device}.")

        self._axis[:3] = prismatic_axis.view(-1, 1)

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        return angle @ self.axis[:3].view(1, -1)


class _PrismaticJointXYZImpl(_PrismaticJointImpl):
    def __init__(
        self,
        name: str,
        axis_id: int,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, parent_link, child_link, origin, dtype, device)
        if axis_id < 0 or axis_id >= 3:
            raise ValueError("The axis_id must be one of (0, 1, 2).")
        self._axis_id = axis_id
        self._axis[axis_id] = 1

    @property
    def axis_id(self) -> int:
        return self._axis_id

    def _translation_impl(self, angle: torch.Tensor) -> torch.Tensor:
        trans = angle.new_zeros(angle.shape[0], 3)
        trans[:, self.axis_id] = angle
        return trans

    def _relative_pose_impl(self, angle: torch.Tensor) -> torch.Tensor:
        _PrismaticJointXYZImpl._check_input(angle)
        angle = angle.view(-1, 1)
        ret = angle.new_empty(angle.shape[0], 3, 4)
        ret[:, :, :3] = self.origin[:, :, :3]
        ret[:, :, 3] = angle * self.origin[:, :, self.axis_id] + self.origin[:, :, 3]
        return ret


class PrismaticJointX(_PrismaticJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, 0, parent_link, child_link, origin, dtype, device)


class PrismaticJointY(_PrismaticJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, 1, parent_link, child_link, origin, dtype, device)


class PrismaticJointZ(_PrismaticJointXYZImpl):
    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        parent_link: Optional[Link] = None,
        child_link: Optional[Link] = None,
        origin: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
    ):
        super().__init__(name, id, 2, parent_link, child_link, origin, dtype, device)
