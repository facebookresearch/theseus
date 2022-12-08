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


def test_sparse_linearization():

    objective, ordering, A, b = build_test_objective_and_linear_system()

    linearization = th.SparseLinearization(objective, ordering=ordering)
    linearization.linearize()

    assert b.isclose(linearization.b).all()

    batch_size = A.shape[0]

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :])
        assert A[i, :, :].isclose(torch.Tensor(csrAi.todense())).all()

    for i in range(batch_size):
        assert b[i].isclose(linearization.b[i]).all()

    # Test Atb result
    atb_expected = A.transpose(1, 2).bmm(b.unsqueeze(2))
    atb_out = linearization.Atb
    torch.testing.assert_close(atb_expected, atb_out)

    # Test Av() with a random v
    rng = torch.Generator()
    rng.manual_seed(1009)
    for _ in range(20):
        v = torch.randn(A.shape[0], A.shape[2], 1)
        av_expected = A.bmm(v).squeeze(2)
        av_out = linearization.Av(v.squeeze(2))
        torch.testing.assert_close(av_expected, av_out)
