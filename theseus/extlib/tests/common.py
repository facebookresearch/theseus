# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

_BASPACHO_NOT_INSTALLED_MSG = "Baspacho solver not in theseus extension library."


def run_if_baspacho():
    # Not sure what's a better place to put this in
    import pytest

    try:
        import theseus.extlib.baspacho_solver  # noqa: F401

        BASPACHO_EXT_NOT_AVAILABLE = False
    except ModuleNotFoundError:
        BASPACHO_EXT_NOT_AVAILABLE = True

    return pytest.mark.skipif(
        BASPACHO_EXT_NOT_AVAILABLE, reason=_BASPACHO_NOT_INSTALLED_MSG
    )
