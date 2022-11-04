# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch  # needed for import of Torch C++ extensions to work
from scipy.sparse import csr_matrix, tril


from tests.extlib.common import run_if_baspacho
from theseus.utils import random_sparse_binary_matrix, split_into_param_sizes


def check_baspacho(
    batch_size,
    rows_to_cols_ratio,
    num_cols,
    param_size_range,
    fill,
    dev="cpu",
    verbose=False,
):
    rng = torch.Generator(device=dev)
    rng.manual_seed(hash(str([batch_size, rows_to_cols_ratio, num_cols, fill])))

    # this is necessary assumption, so that the hessian can be full rank. actually we
    # add some damping to At*A's diagonal, so not really necessary
    assert rows_to_cols_ratio >= 1.0
    num_rows = round(rows_to_cols_ratio * num_cols)

    if isinstance(param_size_range, str):
        param_size_range = [int(x) for x in param_size_range.split(":")]

    from theseus.extlib.baspacho_solver import SymbolicDecomposition

    A_skel = random_sparse_binary_matrix(
        num_rows, num_cols, fill, min_entries_per_col=1, rng=rng
    )
    A_num_cols = num_cols
    A_rowPtr = torch.tensor(A_skel.indptr, dtype=torch.int64).to(dev)
    A_colInd = torch.tensor(A_skel.indices, dtype=torch.int64).to(dev)
    A_num_rows = A_rowPtr.size(0) - 1
    A_nnz = A_colInd.size(0)
    A_val = torch.rand(
        (batch_size, A_nnz), device=dev, dtype=torch.double, generator=rng
    )
    b = torch.rand(
        (batch_size, A_num_rows), device=dev, dtype=torch.double, generator=rng
    )

    paramSizes = split_into_param_sizes(
        num_cols, param_size_range[0], param_size_range[1], rng
    )
    nParams = len(paramSizes)
    paramStarts = np.cumsum([0, *paramSizes])
    to_blocks = csr_matrix(
        (np.ones(num_cols), np.arange(num_cols), paramStarts), (nParams, num_cols)
    )
    A_blk = A_skel @ to_blocks.T

    # diagonal automatically assumed to be filled
    AtA_blk = (tril(A_blk.T @ A_blk)).tocsr()

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

    s = SymbolicDecomposition(
        torch.tensor(paramSizes, dtype=torch.int64),
        torch.tensor(AtA_blk.indptr, dtype=torch.int64),
        torch.tensor(AtA_blk.indices, dtype=torch.int64),
        dev,
    )
    f = s.create_numeric_decomposition(batch_size)

    f.add_MtM(A_val, A_rowPtr, A_colInd)
    beta = 0.01 * torch.rand(batch_size, device=dev, dtype=torch.double, generator=rng)
    alpha = torch.rand(batch_size, device=dev, dtype=torch.double, generator=rng)
    f.damp(alpha, beta)
    f.factor()

    b = torch.rand(
        (batch_size, A_num_rows), device=dev, dtype=torch.double, generator=rng
    )
    Atb = torch.tensor(
        np.array([A_csr[i].T @ b[i].cpu().numpy() for i in range(batch_size)])
    ).to(dev)
    if verbose:
        print("Atb[0]:", Atb[0])

    sol = Atb.clone()
    f.solve(sol)

    damp_diags = [
        alpha[i].item() * np.diag(np.diag(AtA_csr[i].todense()))
        for i in range(batch_size)
    ]
    residuals = [
        (AtA_csr[i] + damp_diags[i]) @ sol[i].cpu().numpy()
        + beta[i].item() * sol[i].cpu().numpy()
        - Atb[i].cpu().numpy()
        for i in range(batch_size)
    ]
    if verbose:
        print("residuals:", [np.linalg.norm(res) for res in residuals])

    assert all(np.linalg.norm(res) < 1e-10 for res in residuals)


@run_if_baspacho()
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.1, 1.7])
@pytest.mark.parametrize("num_cols", [30, 70])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13"])
@pytest.mark.parametrize("fill", [0.02, 0.05])
def test_baspacho_cpu(batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill):
    check_baspacho(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cpu",
    )


@run_if_baspacho()
@pytest.mark.cudaext
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.1, 1.7])
@pytest.mark.parametrize("num_cols", [30, 70])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13"])
@pytest.mark.parametrize("fill", [0.02, 0.05])
def test_baspacho_cuda(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_baspacho(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cuda",
    )
