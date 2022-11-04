# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch
from torch.autograd import gradcheck

import theseus as th


def _build_sparse_mat(batch_size):
    torch.manual_seed(37)
    all_cols = list(range(10))
    col_ind = []
    row_ptr = [0]
    for i in range(12):
        start = max(0, i - 2)
        end = min(i + 1, 10)
        col_ind += all_cols[start:end]
        row_ptr.append(len(col_ind))
    data = torch.randn(size=(batch_size, len(col_ind)), dtype=torch.double)
    return 12, 10, data, col_ind, row_ptr


@pytest.mark.cudaext
def test_sparse_backward_step():
    if not torch.cuda.is_available():
        return
    from theseus.optimizer.autograd import LUCudaSolveFunction

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.LUCudaSparseSolver(
        void_objective, linearization_kwargs={"ordering": void_ordering}, damping=0.01
    )
    linearization = solver.linearization

    batch_size = 4
    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_val = data.cuda()
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr
    linearization.b = torch.randn(
        size=(batch_size, num_rows), dtype=torch.double
    ).cuda()

    linearization.A_val.requires_grad = True
    linearization.b.requires_grad = True
    # Only need this line for the test since the objective is a mock
    solver.reset(batch_size=batch_size)
    damping_alpha_beta = (
        0.5 * torch.rand_like(linearization.A_val[:, 0]),
        1.3 * torch.rand_like(linearization.A_val[:, 0]),
    )
    inputs = (
        linearization.A_val,
        linearization.b,
        linearization.structure(),
        solver.A_rowPtr,
        solver.A_colInd,
        solver._solver_contexts[solver._last_solver_context],
        damping_alpha_beta,
        False,  # it's the same matrix, so no overwrite problems
    )

    assert gradcheck(LUCudaSolveFunction.apply, inputs, eps=3e-4, atol=1e-3)
