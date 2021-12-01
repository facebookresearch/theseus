# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401

import theseus as th

from .common import run_nonlinear_least_squares_check


def test_levenberg_marquartd():
    for ellipsoidal_damping in [False]:
        for damping in [0, 0.001, 0.01, 0.1]:
            run_nonlinear_least_squares_check(
                th.LevenbergMarquardt,
                {
                    "damping": damping,
                    "ellipsoidal_damping": ellipsoidal_damping,
                    "damping_eps": 0.0,
                },
                singular_check=damping < 0.001,
            )
