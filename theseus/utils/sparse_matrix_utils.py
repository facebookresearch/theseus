# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple

import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix


def _mat_vec_cpu(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v):
    assert batch_size == A_val.shape[0]
    num_rows = len(A_rowPtr) - 1
    retv_data = np.array(
        [
            csr_matrix((A_val[i].numpy(), A_colInd, A_rowPtr), (num_rows, num_cols))
            * v[i]
            for i in range(batch_size)
        ],
        dtype=np.float64,
    )
    return torch.tensor(retv_data, dtype=torch.float64)


def mat_vec(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v):
    if A_rowPtr.device.type == "cuda":
        try:
            from theseus.extlib.mat_mult import mat_vec as mat_vec_cuda
        except Exception as e:
            raise RuntimeError(
                "Theseus C++/Cuda extension cannot be loaded\n"
                "even if Cuda appears to be available. Make sure Theseus\n"
                "is installed with Cuda support (export CUDA_HOME=...)\n"
                f"{type(e).__name__}: {e}"
            )
        return mat_vec_cuda(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v)
    else:
        return _mat_vec_cpu(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v)


def _tmat_vec_cpu(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v):
    assert batch_size == A_val.shape[0]
    num_rows = len(A_rowPtr) - 1
    retv_data = np.array(
        [
            csc_matrix((A_val[i].numpy(), A_colInd, A_rowPtr), (num_cols, num_rows))
            * v[i]
            for i in range(batch_size)
        ],
        dtype=np.float64,
    )
    return torch.tensor(retv_data, dtype=torch.float64)


def tmat_vec(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v):
    if A_rowPtr.device.type == "cuda":
        try:
            from theseus.extlib.mat_mult import tmat_vec as tmat_vec_cuda
        except Exception as e:
            raise RuntimeError(
                "Theseus C++/Cuda extension cannot be loaded\n"
                "even if Cuda appears to be available. Make sure Theseus\n"
                "is installed with Cuda support (export CUDA_HOME=...)\n"
                f"{type(e).__name__}: {e}"
            )
        return tmat_vec_cuda(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v)
    else:
        return _tmat_vec_cpu(batch_size, num_cols, A_rowPtr, A_colInd, A_val, v)


def random_sparse_binary_matrix(
    rows: int, cols: int, fill: float, min_entries_per_col: int, rng: torch.Generator
) -> csr_matrix:
    retv = lil_matrix((rows, cols))

    if min_entries_per_col > 0:
        min_entries_per_col = min(rows, min_entries_per_col)
        rows_array = torch.arange(rows, device=rng.device)
        rows_array_f = rows_array.to(dtype=torch.float)
        for c in range(cols):
            row_selection = rows_array[
                rows_array_f.multinomial(min_entries_per_col, generator=rng)
            ].cpu()
            for r in row_selection:
                retv[r, c] = 1.0

    # make sure last row is non-empty, so: len(indptr) = rows+1
    retv[rows - 1, int(torch.randint(cols, (), device=rng.device, generator=rng))] = 1.0

    num_entries = int(fill * rows * cols)
    while retv.getnnz() < num_entries:
        col = int(torch.randint(cols, (), device=rng.device, generator=rng))
        row = int(torch.randint(rows, (), device=rng.device, generator=rng))
        retv[row, col] = 1.0

    return retv.tocsr()


def random_sparse_matrix(
    batch_size: int,
    num_rows: int,
    num_cols: int,
    fill: float,
    min_entries_per_col: int,
    rng: torch.Generator,
    device: torch.device,
    int_dtype: torch.dtype = torch.int64,
    float_dtype: torch.dtype = torch.double,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    A_skel = random_sparse_binary_matrix(
        num_rows, num_cols, fill, min_entries_per_col=min_entries_per_col, rng=rng
    )
    A_row_ptr = torch.tensor(A_skel.indptr, dtype=int_dtype).to(device)
    A_col_ind = torch.tensor(A_skel.indices, dtype=int_dtype).to(device)
    A_val = torch.rand(
        batch_size,
        A_col_ind.size(0),
        device=rng.device,
        dtype=float_dtype,
        generator=rng,
    ).to(device)
    return A_col_ind, A_row_ptr, A_val, A_skel


def split_into_param_sizes(
    n: int, param_size_range_min: int, param_size_range_max: int, rng: torch.Generator
) -> List[int]:
    paramSizes = []
    tot = 0
    while tot < n:
        newParam = min(
            int(
                torch.randint(
                    param_size_range_min,
                    param_size_range_max,
                    (),
                    device=rng.device,
                    generator=rng,
                )
            ),
            n - tot,
        )
        tot += newParam
        paramSizes.append(newParam)
    return paramSizes
