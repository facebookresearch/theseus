# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch  # needed for import of Torch C++ extensions to work
from scipy.sparse import csr_matrix, tril

from theseus.utils import random_sparse_binary_matrix

# ideally we would like to support batch_size <= init_batch_size, but
# because of limitations of cublas those have to be always identical
def check_baspacho(
    init_batch_size, batch_size, num_rows, num_cols, fill, dev="cpu", verbose=False
):
    # this is necessary assumption, so that the hessian is full rank
    assert num_rows >= num_cols

    from theseus.extlib.baspacho_solver import SymbolicDecomposition
    
    A_skel = random_sparse_binary_matrix(
        num_rows, num_cols, fill, min_entries_per_col=3
    )
    A_num_cols = num_cols
    A_rowPtr = torch.tensor(A_skel.indptr, dtype=torch.int64).to(dev)
    A_colInd = torch.tensor(A_skel.indices, dtype=torch.int64).to(dev)
    A_num_rows = A_rowPtr.size(0) - 1
    A_nnz = A_colInd.size(0)
    A_val = torch.rand((batch_size, A_nnz), dtype=torch.double).to(dev)
    b = torch.rand((batch_size, A_num_rows), dtype=torch.double).to(dev)

    paramSizes = []
    tot = 0
    while tot < num_cols:
        newParam = min(np.random.randint(2, 5), num_cols - tot)
        tot += newParam
        paramSizes.append(newParam)
    nParams = len(paramSizes)
    paramStarts = np.cumsum([0, *paramSizes])
    toBlocks = csr_matrix(
            (np.ones(num_cols), np.arange(num_cols), paramStarts),
            (nParams, num_cols)
        )
    A_blk = A_skel @ toBlocks.T
    AtA = tril(A_skel.T @ A_skel)
    AtA_blk = tril(A_blk.T @ A_blk).tocsr()

    A_csr = [
        csr_matrix(
            (A_val[i].cpu(), A_colInd.cpu(), A_rowPtr.cpu()), (A_num_rows, A_num_cols)
        )
        for i in range(batch_size)
    ]
    if verbose:
        print("A[0]:\n", A_csr[0].todense())
        print("b[0]:\n", b[0])

    AtA_csr = [(a.T @ a).tocsr() for a in A_csr]

    if verbose:
        print("AtA[0]:\n", AtA_csr[0].todense())

    s = SymbolicDecomposition(torch.tensor(paramSizes, dtype=torch.int64),
                              torch.tensor(AtA_blk.indptr, dtype=torch.int64),
                              torch.tensor(AtA_blk.indices, dtype=torch.int64))
    f = s.newNumericDecomposition(batch_size)

    f.add_MtM(A_val, A_rowPtr, A_colInd)
    f.factor()

    b = torch.rand((batch_size, A_num_rows), dtype=torch.double).to(dev)
    Atb = torch.tensor(
        np.array([A_csr[i].T @ b[i].cpu().numpy() for i in range(batch_size)])
    ).to(dev)
    if verbose:
        print("Atb[0]:", Atb[0])

    sol = Atb.clone()
    f.solve(sol)

    residuals = [
        AtA_csr[i] @ sol[i].cpu().numpy() - Atb[i].cpu().numpy()
        for i in range(batch_size)
    ]
    if verbose:
        print("residual[0]:", residuals[0])

    assert all(np.linalg.norm(res) < 1e-10 for res in residuals)


def test_lu_solver_1():
    check_baspacho(init_batch_size=5, batch_size=5, num_rows=50, num_cols=30, fill=0.2)


def test_lu_solver_2():
    check_baspacho(
        init_batch_size=5, batch_size=5, num_rows=150, num_cols=60, fill=0.2
    )


def test_lu_solver_3():
    check_baspacho(
        init_batch_size=10, batch_size=10, num_rows=300, num_cols=90, fill=0.2
    )


def test_lu_solver_4():
    check_baspacho(init_batch_size=5, batch_size=5, num_rows=50, num_cols=30, fill=0.1)


def test_lu_solver_5():
    check_baspacho(
        init_batch_size=5, batch_size=5, num_rows=150, num_cols=60, fill=0.1
    )


def test_lu_solver_6():
    check_baspacho(
        init_batch_size=10, batch_size=10, num_rows=300, num_cols=90, fill=0.1
    )


# would like to test when irregular batch_size < init_batch_size,
# but this is currently not supported by cublas, maybe in the future
# def test_lu_solver_7():
#     check_baspacho(init_batch_size=10, batch_size=5, num_rows=150, num_cols=60, fill=0.2)
