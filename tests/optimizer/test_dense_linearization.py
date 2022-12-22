# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch  # noqa: F401

import theseus as th
from tests.optimizer.linearization_test_utils import (
    build_test_objective_and_linear_system,
)


def test_dense_linearization():
    objective, ordering, A, b = build_test_objective_and_linear_system()

    linearization = th.DenseLinearization(objective, ordering=ordering)
    linearization.linearize()

    assert b.isclose(linearization.b).all()

    assert A.isclose(linearization.A).all()

    batch_size = A.shape[0]

    for i in range(batch_size):
        ata = A[i].t() @ A[i]
        atb = (A[i].t() @ b[i]).unsqueeze(1)
        assert ata.isclose(linearization.AtA[i]).all()
        assert atb.isclose(linearization.Atb[i]).all()
