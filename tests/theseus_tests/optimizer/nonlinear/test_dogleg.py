# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401

import theseus as th

from theseus.constants import __FROM_THESEUS_LAYER_TOKEN__
from tests.theseus_tests.optimizer.nonlinear.common import (
    run_nonlinear_least_squares_check,
)


def test_dogleg():
    run_nonlinear_least_squares_check(
        th.Dogleg,
        {__FROM_THESEUS_LAYER_TOKEN__: True},
    )
