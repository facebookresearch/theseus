# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from typing import Callable

import torch


class _LieGroupCheckContext:
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "check_lie_group"):
            cls.contexts.check_lie_group = False
        return cls.contexts.check_lie_group

    @classmethod
    def set_context(cls, check_lie_group: bool):
        cls.contexts.check_lie_group = check_lie_group


class enable_checks(_LieGroupCheckContext):
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        self.prev = _LieGroupCheckContext.get_context()
        _LieGroupCheckContext.set_context(True)

    def __exit__(self, typ, value, traceback) -> None:
        _LieGroupCheckContext.set_context(self.prev)


@torch.no_grad()
def checks_base(tensor: torch.Tensor, check_impl: Callable[[torch.Tensor], None]):
    if not _LieGroupCheckContext.get_context():
        return
    if torch._C._functorch.is_batchedtensor(tensor):
        raise RuntimeError("Lie group checks must be turned off to run with vmap.")
    check_impl(tensor)
