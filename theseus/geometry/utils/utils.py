# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading

import torch


class LieGroupContext(object):
    contexts = threading.local()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.contexts, "use_lie_tangent"):
            cls.contexts.use_lie_tangent = False
        return cls.contexts.use_lie_tangent

    @classmethod
    def set_contexts(cls, use_lie_tangent: bool):
        cls.contexts.use_lie_tangent = use_lie_tangent


class lie_tangent(LieGroupContext):
    def __enter__(self):
        self.prev = super().get_contexts()
        super().set_contexts(True)
        return self

    def __exit__(self, typ, value, traceback):
        super().set_contexts(self.prev)


class no_lie_tangent(LieGroupContext):
    def __enter__(self):
        self.prev = super().get_contexts()
        super().set_contexts(False)
        return self

    def __exit__(self, typ, value, traceback):
        super().set_contexts(self.prev)


class LieGroupTensor(torch.Tensor):
    from torch._C import _disabled_torch_function_impl

    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, group):
        return torch.Tensor._make_subclass(cls, group.data)

    def __init__(self, group):
        self.group_cls = type(group)

    def add_(self, update, alpha=1):
        if LieGroupContext.get_contexts:
            group = self.group_cls(data=self.data)
            grad = group.project(update)
            self.set_(group.retract(alpha * grad).data)
        else:
            self.add_(update, alpha=alpha)

        return self

    def addcdiv_(self, tensor1, tensor2, value=1):
        return (
            self.add_(value * tensor1 / tensor2)
            if LieGroupContext.get_contexts()
            else super().addcdiv_(tensor1, tensor2, value=value)
        )

    def addcmul_(self, tensor1, tensor2, value=1):
        return (
            self.add_(value * tensor1 * tensor2)
            if LieGroupContext.get_contexts()
            else super().addcmul_(tensor1, tensor2, value=value)
        )
