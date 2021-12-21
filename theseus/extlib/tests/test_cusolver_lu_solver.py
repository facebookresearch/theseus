# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch  # needed for import of Torch C++ extensions to work
from scipy.sparse import csr_matrix

from theseus.utils import generate_mock_sparse_matrix


def check_lu_solver(base_batch_size, batch_size, rows, cols, fill, verbose=False):
    # this is necessary assumption, so that the hessian is full rank
    assert rows > cols

    if not torch.cuda.is_available():
        return
    from theseus.extlib.cusolver_lu_solver import CusolverLUSolver

    A_skel = generate_mock_sparse_matrix(rows, cols, fill, min_entries_per_col=3)
    A_cols = cols
    A_rowPtr = torch.tensor(A_skel.indptr, dtype=torch.int).cuda()
    A_colInd = torch.tensor(A_skel.indices, dtype=torch.int).cuda()
    A_rows = A_rowPtr.size(0) - 1
    A_nnz = A_colInd.size(0)
    A_val = torch.rand((batch_size, A_nnz), dtype=torch.double).cuda()
    b = torch.rand((batch_size, A_rows), dtype=torch.double).cuda()

    A_csr = [
        csr_matrix((A_val[i].cpu(), A_colInd.cpu(), A_rowPtr.cpu()), (A_rows, A_cols))
        for i in range(batch_size)
    ]
    if verbose:
        print("A[0]:\n", A_csr[0].todense())
        print("b[0]:\n", b[0])

    AtA_csr = [(a.T @ a).tocsr() for a in A_csr]
    AtA_rowPtr = torch.tensor(AtA_csr[0].indptr).cuda()
    AtA_colInd = torch.tensor(AtA_csr[0].indices).cuda()
    AtA_val = torch.tensor(np.array([m.data for m in AtA_csr])).cuda()
    AtA_rows = AtA_rowPtr.size(0) - 1
    AtA_cols = AtA_rows
    AtA_nnz = AtA_colInd.size(0)  # noqa: F841

    if verbose:
        print("AtA[0]:\n", AtA_csr[0].todense())

    slv = CusolverLUSolver(base_batch_size, AtA_cols, AtA_rowPtr, AtA_colInd)
    singularities = slv.factor(AtA_val)

    if verbose:
        print("singularities:", singularities)

    b = torch.rand((batch_size, A_rows), dtype=torch.double).cuda()
    Atb = torch.tensor(
        np.array([A_csr[i].T @ b[i].cpu().numpy() for i in range(batch_size)])
    ).cuda()
    if verbose:
        print("Atb[0]:", Atb[0])

    sol = Atb.clone()
    slv.solve(sol)
    if verbose:
        print("x[0]:", sol[0])

    residuals = [
        AtA_csr[i] @ sol[i].cpu().numpy() - Atb[i].cpu().numpy()
        for i in range(batch_size)
    ]
    if verbose:
        print("residual[0]:", residuals[0])

    assert all(np.linalg.norm(res) < 1e-10 for res in residuals)


def test_lu_solver_1():
    check_lu_solver(base_batch_size=5, batch_size=5, rows=50, cols=30, fill=0.2)


def test_lu_solver_2():
    check_lu_solver(base_batch_size=5, batch_size=5, rows=150, cols=60, fill=0.2)


def test_lu_solver_3():
    check_lu_solver(base_batch_size=10, batch_size=10, rows=300, cols=90, fill=0.2)


def test_lu_solver_4():
    check_lu_solver(base_batch_size=5, batch_size=5, rows=50, cols=30, fill=0.1)


def test_lu_solver_5():
    check_lu_solver(base_batch_size=5, batch_size=5, rows=150, cols=60, fill=0.1)


def test_lu_solver_6():
    check_lu_solver(base_batch_size=10, batch_size=10, rows=300, cols=90, fill=0.1)


# not yet, the reason is most probably a cublas bug?!
# def test_lu_solver_7():
#     check_lu_solver(base_batch_size=10, batch_size=5, rows=150, cols=60, fill=0.2)
