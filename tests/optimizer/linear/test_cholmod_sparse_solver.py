# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch
from sksparse.cholmod import analyze_AAt

import theseus as th


def _build_sparse_mat(batch_size):
    all_cols = list(range(10))
    col_ind = []
    row_ptr = [0]
    for i in range(12):
        start = max(0, i - 2)
        end = min(i + 1, 10)
        col_ind += all_cols[start:end]
        row_ptr.append(len(col_ind))
    data = torch.randn((batch_size, len(col_ind)))
    return 12, 10, data, col_ind, row_ptr


def test_sparse_solver():

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    damping = 0.2  # set big value for checking
    solver = th.CholmodSparseSolver(
        void_objective,
        linearization_kwargs={"ordering": void_ordering},
        damping=damping,
    )
    linearization = solver.linearization

    batch_size = 4
    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_val = data
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr
    linearization.b = torch.randn((batch_size, num_rows))
    # Only need this line for the test since the objective is a mock
    solver._symbolic_cholesky_decomposition = analyze_AAt(
        linearization.structure().mock_csc_transpose()
    )

    solved_x = solver.solve()

    # also check that damping is being overridden via kwargs correctly
    other_damping = 0.3
    solved_x_other_damping = solver.solve(damping=other_damping)

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :])
        Ai = torch.Tensor(csrAi.todense())
        ata = Ai.T @ Ai
        b = linearization.b[i]
        atb = torch.Tensor(csrAi.transpose() @ b)

        def _check_correctness(solved_x_, damping_):
            # the linear system solved is with matrix (AtA + damping*I)
            atb_check = ata @ solved_x_[i] + damping_ * solved_x_[i]
            max_offset = torch.norm(atb - atb_check, p=float("inf"))
            assert max_offset < 1e-4

        _check_correctness(solved_x, damping)
        _check_correctness(solved_x_other_damping, other_damping)
