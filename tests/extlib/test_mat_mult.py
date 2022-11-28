# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch  # needed for import of Torch C++ extensions to work
from scipy.sparse import csr_matrix

from theseus.utils import random_sparse_matrix


def check_mat_mult(batch_size, num_rows, num_cols, fill, verbose=False):
    if not torch.cuda.is_available():
        return
    from theseus.extlib.mat_mult import apply_damping, mat_vec, mult_MtM, tmat_vec

    rng = torch.Generator()
    rng.manual_seed(0)
    A_col_ind, A_row_ptr, A_val, _ = random_sparse_matrix(
        batch_size, num_rows, num_cols, fill, 3, rng, "cuda:0", int_dtype=torch.int
    )
    A_num_cols = num_cols
    A_num_rows = A_row_ptr.size(0) - 1

    A_csr = [
        csr_matrix(
            (A_val[i].cpu(), A_col_ind.cpu(), A_row_ptr.cpu()), (A_num_rows, A_num_cols)
        )
        for i in range(batch_size)
    ]
    if verbose:
        print("A[0]:\n", A_csr[0].todense())

    # test At * A
    AtA_csr = [(a.T @ a).tocsr() for a in A_csr]
    AtA_row_ptr = torch.tensor(AtA_csr[0].indptr).cuda()
    AtA_col_ind = torch.tensor(AtA_csr[0].indices).cuda()
    AtA_val = torch.tensor(np.array([m.data for m in AtA_csr])).cuda()
    AtA_num_rows = AtA_row_ptr.size(0) - 1
    AtA_num_cols = AtA_num_rows

    if verbose:
        print("\nAtA[0]:\n", AtA_csr[0].todense())

    res = mult_MtM(batch_size, A_row_ptr, A_col_ind, A_val, AtA_row_ptr, AtA_col_ind)
    if verbose:
        print(
            "res[0]:\n",
            csr_matrix(
                (res[0].cpu(), AtA_col_ind.cpu(), AtA_row_ptr.cpu()),
                (AtA_num_rows, AtA_num_cols),
            ).todense(),
        )

    assert AtA_val.isclose(res, atol=1e-10).all()

    # test damping
    old_diagonals = torch.tensor(
        np.array(
            [
                csr_matrix(
                    (res[x].cpu(), AtA_col_ind.cpu(), AtA_row_ptr.cpu()),
                    (AtA_num_rows, AtA_num_cols),
                ).diagonal()
                for x in range(batch_size)
            ]
        )
    )
    alpha = 0.3 * torch.rand(batch_size, dtype=torch.double).cuda()
    beta = 0.7 * torch.rand(batch_size, dtype=torch.double).cuda()
    apply_damping(batch_size, AtA_num_cols, AtA_row_ptr, AtA_col_ind, res, alpha, beta)
    new_diagonals = torch.tensor(
        np.array(
            [
                csr_matrix(
                    (res[x].cpu(), AtA_col_ind.cpu(), AtA_row_ptr.cpu()),
                    (AtA_num_rows, AtA_num_cols),
                ).diagonal()
                for x in range(batch_size)
            ]
        )
    )
    assert new_diagonals.isclose(
        old_diagonals * (1 + alpha.cpu().view(-1, 1)) + beta.cpu().view(-1, 1),
        atol=1e-10,
    ).all()

    # test A * b
    v = torch.rand((batch_size, A_num_cols), dtype=torch.double).cuda()
    A_v = torch.tensor(
        np.array([A_csr[i] @ v[i].cpu() for i in range(batch_size)])
    ).cuda()

    A_v_test = mat_vec(batch_size, A_num_cols, A_row_ptr, A_col_ind, A_val, v)

    if verbose:
        print("A_v:", A_v)
        print("A_v_test:", A_v_test)

    assert A_v.isclose(A_v_test, atol=1e-10).all()

    # test At * b
    w = torch.rand((batch_size, A_num_rows), dtype=torch.double).cuda()
    At_w = torch.tensor(
        np.array([A_csr[i].T @ w[i].cpu() for i in range(batch_size)])
    ).cuda()

    At_w_test = tmat_vec(batch_size, A_num_cols, A_row_ptr, A_col_ind, A_val, w)

    if verbose:
        print("A_w:", At_w)
        print("A_w_test:", At_w_test)

    assert At_w.isclose(At_w_test, atol=1e-10).all()


@pytest.mark.cudaext
def test_mat_mult_1():
    check_mat_mult(batch_size=5, num_rows=50, num_cols=30, fill=0.2)


@pytest.mark.cudaext
def test_mat_mult_2():
    check_mat_mult(batch_size=5, num_rows=150, num_cols=60, fill=0.2)


@pytest.mark.cudaext
def test_mat_mult_3():
    check_mat_mult(batch_size=10, num_rows=300, num_cols=90, fill=0.2)


@pytest.mark.cudaext
def test_mat_mult_4():
    check_mat_mult(batch_size=5, num_rows=50, num_cols=30, fill=0.1)


@pytest.mark.cudaext
def test_mat_mult_5():
    check_mat_mult(batch_size=5, num_rows=150, num_cols=60, fill=0.1)


@pytest.mark.cudaext
def test_mat_mult_6():
    check_mat_mult(batch_size=10, num_rows=300, num_cols=90, fill=0.1)
