# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from typing import Any

import torch

from .lie_group import LieGroup


class _LieGroupContext(object):
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "use_lie_tangent"):
            cls.contexts.use_lie_tangent = False
        return cls.contexts.use_lie_tangent

    @classmethod
    def set_context(cls, use_lie_tangent: bool):
        cls.contexts.use_lie_tangent = use_lie_tangent


class set_lie_tangent_enabled(object):
    def __init__(self, mode: bool) -> None:
        self.prev = _LieGroupContext.get_context()
        _LieGroupContext.set_context(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _LieGroupContext.set_context(self.prev)


class enable_lie_tangent(object):
    def __enter__(self) -> None:
        self.prev = _LieGroupContext.get_context()
        _LieGroupContext.set_context(True)

    def __exit__(self, typ, value, traceback) -> None:
        _LieGroupContext.set_context(self.prev)


class no_lie_tangent(_LieGroupContext):
    def __enter__(self):
        self.prev = super().get_context()
        _LieGroupContext.set_context(False)
        return self

    def __exit__(self, typ, value, traceback):
        _LieGroupContext.set_context(self.prev)


class LieGroupTensor(torch.Tensor):
    from torch._C import _disabled_torch_function_impl  # type: ignore

    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, group):
        return torch.Tensor._make_subclass(cls, group.tensor)

    def __init__(self, group: LieGroup):
        self.group_cls = type(group)

    def add_(self, update, alpha=1):
        if _LieGroupContext.get_context():
            group = self.group_cls(tensor=self.data)
            grad = group.project(update)
            self.set_(group.retract(alpha * grad).tensor)
        else:
            self.add_(update, alpha=alpha)

        return self

    def addcdiv_(self, tensor1, tensor2, value=1):
        self.add_(
            value * tensor1 / tensor2
        ) if _LieGroupContext.get_context() else super().addcdiv_(
            tensor1, tensor2, value=value
        )
        return self

    def addcmul_(self, tensor1, tensor2, value=1):
        self.add_(
            value * tensor1 * tensor2
        ) if _LieGroupContext.get_context() else super().addcmul_(
            tensor1, tensor2, value=value
        )
        return self
