# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

_BASPACHO_NOT_INSTALLED_MSG = "Baspacho solver not in theseus extension library."
_LABS_NOT_INSTALLED_MSG = "Theseus Labs is not available."


def _run_if(import_fn, msg):
    import pytest

    try:
        import_fn()

        is_available = False
    except ModuleNotFoundError:
        is_available = True

    return pytest.mark.skipif(is_available, reason=msg)


def run_if_baspacho():
    def _import_fn():
        import theseus.extlib.baspacho_solver  # noqa: F401

    return _run_if(_import_fn, _BASPACHO_NOT_INSTALLED_MSG)


def run_if_labs():
    def _import_fn():
        import theseus.labs  # noqa: F401

    return _run_if(_import_fn, _LABS_NOT_INSTALLED_MSG)
