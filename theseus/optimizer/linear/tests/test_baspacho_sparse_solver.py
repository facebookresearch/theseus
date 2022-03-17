# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

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
    data = torch.randn((batch_size, len(col_ind)), dtype=torch.double)
    return 12, 10, data, col_ind, row_ptr


@pytest.mark.cuda
def test_sparse_solver(dev="cpu"):

    if dev=="cuda" and not torch.cuda.is_available():
        return

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.BaspachoSparseSolver(
        void_objective,
        linearization_kwargs={"ordering": void_ordering},
    )
    linearization = solver.linearization

    batch_size = 4
    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_val = data.to(dev)
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr
    linearization.b = torch.randn((batch_size, num_rows), dtype=torch.double).to(dev)

    # also need: var dims and var_start_cols (because baspacho is blockwise)
    linearization.var_dims = [2,1,3,1,2,1]
    linearization.var_start_cols = [0,2,3,6,7,9]

    # Only need this line for the test since the objective is a mock
    solver.reset(dev=dev)

    solved_x = solver.solve()

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :].cpu())
        Ai = torch.tensor(csrAi.todense(), dtype=torch.double)
        ata = Ai.T @ Ai
        b = linearization.b[i].cpu()
        atb = torch.Tensor(csrAi.transpose() @ b)

        # the linear system solved is with matrix AtA
        atb_check = ata @ solved_x[i].cpu()

        max_offset = torch.norm(atb - atb_check, p=float("inf"))
        assert max_offset < 1e-4
