# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from typing import Any


class _LieGroupCheckContext:
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "check_lie_group"):
            cls.contexts.check_lie_group = True
            cls.contexts.silent = False
        return cls.contexts.check_lie_group, cls.contexts.silent

    @classmethod
    def set_context(cls, check_lie_group: bool, silent: bool):
        if not check_lie_group and not silent:
            print(
                "Warnings for disabled Lie group checks can be turned "
                "off by passing silent=True."
            )
        cls.contexts.check_lie_group = check_lie_group
        cls.contexts.silent = silent


class set_lie_group_check_enabled:
    def __init__(self, mode: bool, silent: bool = False) -> None:
        self.prev = _LieGroupCheckContext.get_context()
        _LieGroupCheckContext.set_context(mode, silent)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _LieGroupCheckContext.set_context(*self.prev)


class enable_lie_group_check(_LieGroupCheckContext):
    def __init__(self, silent: bool = False) -> None:
        self._silent = silent

    def __enter__(self) -> None:
        self.prev = _LieGroupCheckContext.get_context()
        _LieGroupCheckContext.set_context(True, self._silent)

    def __exit__(self, typ, value, traceback) -> None:
        _LieGroupCheckContext.set_context(*self.prev)


class no_lie_group_check(_LieGroupCheckContext):
    def __init__(self, silent: bool = False) -> None:
        self._silent = silent

    def __enter__(self):
        self.prev = super().get_context()
        _LieGroupCheckContext.set_context(False, self._silent)
        return self

    def __exit__(self, typ, value, traceback):
        _LieGroupCheckContext.set_context(*self.prev)
