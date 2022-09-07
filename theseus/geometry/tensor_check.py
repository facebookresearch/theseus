# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from typing import Any


class _LieGroupTensorCheckContext:
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "check_lie_group_tensor"):
            cls.contexts.check_lie_group_tensor = False
        return cls.contexts.check_lie_group_tensor

    @classmethod
    def set_context(cls, check_lie_group_tensor: bool):
        cls.contexts.check_lie_group_tensor = check_lie_group_tensor


class set_lie_group_tensor_check_enabled:
    def __init__(self, mode: bool) -> None:
        self.prev = _LieGroupTensorCheckContext.get_context()
        _LieGroupTensorCheckContext.set_context(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _LieGroupTensorCheckContext.set_context(self.prev)


class enable_lie_group_tensor_check(_LieGroupTensorCheckContext):
    def __enter__(self) -> None:
        self.prev = _LieGroupTensorCheckContext.get_context()
        _LieGroupTensorCheckContext.set_context(True)

    def __exit__(self, typ, value, traceback) -> None:
        _LieGroupTensorCheckContext.set_context(self.prev)


class no_lie_group_tensor_check(_LieGroupTensorCheckContext):
    def __enter__(self):
        self.prev = super().get_context()
        _LieGroupTensorCheckContext.set_context(False)
        return self

    def __exit__(self, typ, value, traceback):
        _LieGroupTensorCheckContext.set_context(self.prev)
