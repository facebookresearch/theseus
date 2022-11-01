# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch  # needed for import of Torch C++ extensions to work
from scipy.sparse import csr_matrix

from theseus.utils import random_sparse_binary_matrix


# ideally we would like to support batch_size <= init_batch_size, but
# because of limitations of cublas those have to be always identical
def check_lu_solver(
    init_batch_size, batch_size, num_rows, num_cols, fill, verbose=False
):
    # this is necessary assumption, so that the hessian is full rank
    assert num_rows >= num_cols

    if not torch.cuda.is_available():
        return
    from theseus.extlib.cusolver_lu_solver import CusolverLUSolver

    rng = torch.Generator()
    rng.manual_seed(0)
    A_skel = random_sparse_binary_matrix(
        num_rows, num_cols, fill, min_entries_per_col=3, rng=rng
    )
    A_num_cols = num_cols
    A_rowPtr = torch.tensor(A_skel.indptr, dtype=torch.int).cuda()
    A_colInd = torch.tensor(A_skel.indices, dtype=torch.int).cuda()
    A_num_rows = A_rowPtr.size(0) - 1
    A_nnz = A_colInd.size(0)
    A_val = torch.rand((batch_size, A_nnz), dtype=torch.double).cuda()
    b = torch.rand((batch_size, A_num_rows), dtype=torch.double).cuda()

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
    AtA_rowPtr = torch.tensor(AtA_csr[0].indptr).cuda()
    AtA_colInd = torch.tensor(AtA_csr[0].indices).cuda()
    AtA_val = torch.tensor(np.array([m.data for m in AtA_csr])).cuda()
    AtA_num_rows = AtA_rowPtr.size(0) - 1
    AtA_num_cols = AtA_num_rows
    AtA_nnz = AtA_colInd.size(0)  # noqa: F841

    if verbose:
        print("AtA[0]:\n", AtA_csr[0].todense())

    slv = CusolverLUSolver(init_batch_size, AtA_num_cols, AtA_rowPtr, AtA_colInd)
    singularities = slv.factor(AtA_val)

    if verbose:
        print("singularities:", singularities)

    b = torch.rand((batch_size, A_num_rows), dtype=torch.double).cuda()
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


@pytest.mark.cudaext
def test_lu_solver_1():
    check_lu_solver(init_batch_size=5, batch_size=5, num_rows=50, num_cols=30, fill=0.2)


@pytest.mark.cudaext
def test_lu_solver_2():
    check_lu_solver(
        init_batch_size=5, batch_size=5, num_rows=150, num_cols=60, fill=0.2
    )


@pytest.mark.cudaext
def test_lu_solver_3():
    check_lu_solver(
        init_batch_size=10, batch_size=10, num_rows=300, num_cols=90, fill=0.2
    )


@pytest.mark.cudaext
def test_lu_solver_4():
    check_lu_solver(init_batch_size=5, batch_size=5, num_rows=50, num_cols=30, fill=0.1)


@pytest.mark.cudaext
def test_lu_solver_5():
    check_lu_solver(
        init_batch_size=5, batch_size=5, num_rows=150, num_cols=60, fill=0.1
    )


@pytest.mark.cudaext
def test_lu_solver_6():
    check_lu_solver(
        init_batch_size=10, batch_size=10, num_rows=300, num_cols=90, fill=0.1
    )


# would like to test when irregular batch_size < init_batch_size,
# but this is currently not supported by cublas, maybe in the future
# def test_lu_solver_7():
#     check_lu_solver(init_batch_size=10, batch_size=5, num_rows=150, num_cols=60, fill=0.2)
