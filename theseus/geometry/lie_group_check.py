# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
from contextlib import contextmanager


class _LieGroupCheckContext:
    contexts = threading.local()

    @classmethod
    def get_context(cls):
        if not hasattr(cls.contexts, "check_lie_group"):
            cls.contexts.check_lie_group = True
            cls.contexts.silent = False
            cls.contexts.silence_internal_warnings = False
        return (
            cls.contexts.check_lie_group,
            cls.contexts.silent,
            cls.contexts.silence_internal_warnings,
        )

    @classmethod
    def set_context(
        cls, check_lie_group: bool, silent: bool, silence_internal_warnings: bool
    ):
        if not check_lie_group and not silent:
            print(
                "Warnings for disabled Lie group checks can be turned "
                "off by passing silent=True."
            )
        cls.contexts.check_lie_group = check_lie_group
        cls.contexts.silent = silent
        cls.contexts.silence_internal_warnings = silence_internal_warnings


@contextmanager
def set_lie_group_check_enabled(
    mode: bool, silent: bool = False, silence_internal_warnings: bool = False
):
    """Sets whether or not Lie group checks are enabled within a context.

    :param check_lie_group: Disables Lie group checks if false.
    :param silent: Disables a warning that Lie group checks are disabled.
    :param silence_internal_warnings: Whether to suppress recoverable
        warning messages during Lie group checks, e.g. when normalization
        is performed automatically.
    """
    prev = _LieGroupCheckContext.get_context()
    _LieGroupCheckContext.set_context(mode, silent, silence_internal_warnings)
    yield
    _LieGroupCheckContext.set_context(*prev)


@contextmanager
def enable_lie_group_check(
    silent: bool = False, silence_internal_warnings: bool = False
):
    """Enables Lie group checks while the context is active.

    :param silent: Disables a warning that Lie group checks are disabled.
    :param silence_internal_warnings: Whether to suppress recoverable
        warning messages during Lie group checks, e.g. when normalization
        is performed automatically.
    """
    with set_lie_group_check_enabled(True, silent, silence_internal_warnings):
        yield


@contextmanager
def no_lie_group_check(silent: bool = False, silence_internal_warnings: bool = False):
    """Disables Lie group checks while the context is active.

    :param silent: Disables a warning that Lie group checks are disabled.
    :param silence_internal_warnings: Whether to suppress recoverable
        warning messages during Lie group checks, e.g. when normalization
        is performed automatically.
    """
    with set_lie_group_check_enabled(False, silent, silence_internal_warnings):
        yield


@contextmanager
def silence_internal_warnings():
    """Silences internal warnings if they would be emitted.

    Code that should be silenced should accomplish silencing via:

        _, _, silence_internal_warnings = _LieGroupCheckContext.get_context()
        if not silence_internal_warnings:
            emit_warning("warning message")
    """
    mode, silent, prev_silence_internal_warnings = _LieGroupCheckContext.get_context()
    _LieGroupCheckContext.set_context(mode, silent, silence_internal_warnings=True)
    yield
    _LieGroupCheckContext.set_context(mode, silent, prev_silence_internal_warnings)
