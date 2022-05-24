import pytest  # noqa: F401
import torch  # noqa: F401

import theseus as th
from theseus.optimizer.tests.linearization_test_utils import (
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
