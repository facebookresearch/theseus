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
        parent: Any = None,
        children: Optional[List[Any]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self._name = name
        self._id = id
        self._parent = parent
        self._children = children if children else []
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    @property
    def parent(self) -> Any:
        return self._parent

    @property
    def children(self) -> List[Any]:
        return self._children

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_id(self, id: int):
        self._id = id

    def set_parent(self, parent: Any):
        self._parent = parent

    def set_children(self, children: List[Any]):
        self._children = children

    def add_child(self, child: Any):
        self._children.append(child)

    def remove_child(self, child: Any):
        self._children.remove(child)
