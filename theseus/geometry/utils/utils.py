# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


class LieGroupTensor(torch.Tensor):
    from torch._C import _disabled_torch_function_impl

    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, group):
        return torch.Tensor._make_subclass(cls, group.data)

    def __init__(self, group):
        self.group_cls = type(group)

    def add_(self, update, alpha=1):
        group = self.group_cls(data=self.data)
        grad = group.project(update)
        self.set_(group.retract(alpha * grad).data)

    def addcdiv_(self, tensor1, tensor2, value=1):
        self.add_(value * tensor1 / tensor2)

    def addcmul_(self, tensor1, tensor2, value=1):
        self.add_(value * tensor1 * tensor2)
