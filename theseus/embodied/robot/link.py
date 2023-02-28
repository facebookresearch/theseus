# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Optional, Any
import torch


class Link(abc.ABC):
    def __init__(
        self,
        name: str,
        id: int = -1,
        parent_joint: Any = None,
        child_joints: Optional[List[Any]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._name = name
        self._id = id
        self._parent_joint = parent_joint
        self._child_joints = child_joints if child_joints else []
        self._ancestor_links: List[Link] = []
        self._ancestor_active_joint_ids: List[int] = []
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
    def parent_joint(self) -> Any:
        return self._parent_joint

    @property
    def child_joints(self) -> List[Any]:
        return self._child_joints

    @property
    def ancestor_links(self) -> List["Link"]:
        return self._ancestor_links

    @property
    def ancestor_active_joint_ids(self):
        return self._ancestor_active_joint_ids

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_id(self, id: int):
        self._id = id

    def set_parent_joint(self, parent_joint: Any):
        self._parent_joint = parent_joint

    def set_child_joints(self, child_joints: List[Any]):
        self._child_joints = child_joints

    def set_ancestor_links(self, ancestor_links: List["Link"]):
        self._ancestor_links = ancestor_links

    def set_ancestor_active_joint_ids(self, ancestor_active_joint_ids: List[int]):
        self._ancestor_active_joint_ids = ancestor_active_joint_ids

    def update_ancestor_links(self):
        curr = self.parent_link
        self._ancestor_links = []
        while curr is not None:
            self._ancestor_links.insert(0, curr)
            curr = curr.parent_link

    def add_child_joint(self, child_joint: Any):
        self._child_joints.append(child_joint)

    def remove_child_joint(self, child_joint: Any):
        self._child_joints.remove(child_joint)
