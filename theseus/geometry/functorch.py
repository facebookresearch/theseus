# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from typing import Any


class _FunctorchContext:
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "use_functorch"):
            cls.contexts.use_functorch = False
        return cls.contexts.use_functorch

    @classmethod
    def set_context(cls, use_functorch: bool):
        cls.contexts.use_functorch = use_functorch


class set_functorch_enabled:
    def __init__(self, mode: bool) -> None:
        self.prev = _FunctorchContext.get_context()
        _FunctorchContext.set_context(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _FunctorchContext.set_context(self.prev)


class enable_functorch(_FunctorchContext):
    def __enter__(self) -> None:
        self.prev = _FunctorchContext.get_context()
        _FunctorchContext.set_context(True)

    def __exit__(self, typ, value, traceback) -> None:
        _FunctorchContext.set_context(self.prev)


class no_functorch(_FunctorchContext):
    def __enter__(self):
        self.prev = super().get_context()
        _FunctorchContext.set_context(False)
        return self

    def __exit__(self, typ, value, traceback):
        _FunctorchContext.set_context(self.prev)
