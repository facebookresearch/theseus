# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix

from theseus.constants import DeviceType


def _mat_vec_cpu(
    batch_size: int,
    num_cols: int,
    A_row_ptr: torch.Tensor,
    A_col_ind: torch.Tensor,
    A_val: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    assert batch_size == A_val.shape[0]
    num_rows = len(A_row_ptr) - 1
    retv_data = np.array(
        [
            csr_matrix((A_val[i].numpy(), A_col_ind, A_row_ptr), (num_rows, num_cols))
            * v[i]
            for i in range(batch_size)
        ],
        dtype=np.float64,
    )
    return torch.tensor(retv_data, dtype=torch.float64)


def mat_vec(
    batch_size: int,
    num_cols: int,
    A_row_ptr: torch.Tensor,
    A_col_ind: torch.Tensor,
    A_val: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    if A_row_ptr.device.type == "cuda":
        try:
            from theseus.extlib.mat_mult import mat_vec as mat_vec_cuda
        except Exception as e:
            raise RuntimeError(
                "Theseus C++/Cuda extension cannot be loaded\n"
                "even if Cuda appears to be available. Make sure Theseus\n"
                "is installed with Cuda support (export CUDA_HOME=...)\n"
                f"{type(e).__name__}: {e}"
            )
        return mat_vec_cuda(batch_size, num_cols, A_row_ptr, A_col_ind, A_val, v)
    else:
        return _mat_vec_cpu(batch_size, num_cols, A_row_ptr, A_col_ind, A_val, v)


def _tmat_vec_cpu(
    batch_size: int,
    num_cols: int,
    A_row_ptr: torch.Tensor,
    A_col_ind: torch.Tensor,
    A_val: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    assert batch_size == A_val.shape[0]
    num_rows = len(A_row_ptr) - 1
    retv_data = np.array(
        [
            csc_matrix((A_val[i].numpy(), A_col_ind, A_row_ptr), (num_cols, num_rows))
            * v[i]
            for i in range(batch_size)
        ],
        dtype=np.float64,
    )
    return torch.tensor(retv_data, dtype=torch.float64)


def tmat_vec(
    batch_size: int,
    num_cols: int,
    A_row_ptr: torch.Tensor,
    A_col_ind: torch.Tensor,
    A_val: torch.Tensor,
    v: torch.Tensor,
):
    if A_row_ptr.device.type == "cuda":
        try:
            from theseus.extlib.mat_mult import tmat_vec as tmat_vec_cuda
        except Exception as e:
            raise RuntimeError(
                "Theseus C++/Cuda extension cannot be loaded\n"
                "even if Cuda appears to be available. Make sure Theseus\n"
                "is installed with Cuda support (export CUDA_HOME=...)\n"
                f"{type(e).__name__}: {e}"
            )
        return tmat_vec_cuda(batch_size, num_cols, A_row_ptr, A_col_ind, A_val, v)
    else:
        return _tmat_vec_cpu(batch_size, num_cols, A_row_ptr, A_col_ind, A_val, v)


def _sparse_mat_vec_fwd_backend(
    ctx: Any,
    num_cols: int,
    A_row_ptr: torch.Tensor,
    A_col_ind: torch.Tensor,
    A_val: torch.Tensor,
    v: torch.Tensor,
    op: Callable[
        [int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> torch.Tensor:
    assert A_row_ptr.ndim == 1
    assert A_col_ind.ndim == 1
    assert A_val.ndim == 2
    assert v.ndim == 2
    ctx.save_for_backward(A_val, A_row_ptr, A_col_ind, v)
    ctx.num_cols = num_cols
    return op(A_val.shape[0], num_cols, A_row_ptr, A_col_ind, A_val, v)


def _sparse_mat_vec_bwd_backend(
    ctx: Any, grad_output: torch.Tensor, is_tmat: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    A_val, A_row_ptr, A_col_ind, v = ctx.saved_tensors
    num_rows = len(A_row_ptr) - 1
    A_grad = torch.zeros_like(A_val)  # (batch_size, nnz)
    v_grad = torch.zeros_like(v)  # (batch_size, num_cols)
    for row in range(num_rows):
        start = A_row_ptr[row]
        end = A_row_ptr[row + 1]
        columns = A_col_ind[start:end].long()
        if is_tmat:
            A_grad[:, start:end] = v[:, row].view(-1, 1) * grad_output[:, columns]
            v_grad[:, row] = (grad_output[:, columns] * A_val[:, start:end]).sum(dim=1)
        else:
            A_grad[:, start:end] = v[:, columns] * grad_output[:, row].view(-1, 1)
            v_grad[:, columns] += grad_output[:, row].view(-1, 1) * A_val[:, start:end]
    return A_grad, v_grad


class _SparseMvPAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        num_cols: int,
        A_row_ptr: torch.Tensor,
        A_col_ind: torch.Tensor,
        A_val: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return _sparse_mat_vec_fwd_backend(
            ctx, num_cols, A_row_ptr, A_col_ind, A_val, v, mat_vec
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[None, None, None, torch.Tensor, torch.Tensor]:
        A_grad, v_grad = _sparse_mat_vec_bwd_backend(ctx, grad_output, False)
        return None, None, None, A_grad, v_grad


class _SparseMtvPAutograd(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        num_cols: int,
        A_row_ptr: torch.Tensor,
        A_col_ind: torch.Tensor,
        A_val: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return _sparse_mat_vec_fwd_backend(
            ctx, num_cols, A_row_ptr, A_col_ind, A_val, v, tmat_vec
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[None, None, None, torch.Tensor, torch.Tensor]:
        A_grad, v_grad = _sparse_mat_vec_bwd_backend(ctx, grad_output, True)
        return None, None, None, A_grad, v_grad


sparse_mv = _SparseMvPAutograd.apply
sparse_mtv = _SparseMtvPAutograd.apply


def random_sparse_binary_matrix(
    num_rows: int,
    num_cols: int,
    fill: float,
    min_entries_per_col: int,
    rng: torch.Generator,
) -> csr_matrix:
    retv = lil_matrix((num_rows, num_cols))

    if num_rows > 1 and min_entries_per_col > 0:
        min_entries_per_col = min(num_rows, min_entries_per_col)
        rows_array = torch.arange(num_rows, device=rng.device)
        rows_array_f = rows_array.to(dtype=torch.float)
        for c in range(num_cols):
            row_selection = rows_array[
                rows_array_f.multinomial(min_entries_per_col, generator=rng)
            ].cpu()
            for r in row_selection:
                retv[r, c] = 1.0

    # make sure last row is non-empty, so: len(indptr) = rows+1
    retv[
        num_rows - 1, int(torch.randint(num_cols, (), device=rng.device, generator=rng))
    ] = 1.0

    num_entries = int(fill * num_rows * num_cols)
    while retv.getnnz() < num_entries:
        col = int(torch.randint(num_cols, (), device=rng.device, generator=rng))
        row = int(torch.randint(num_rows, (), device=rng.device, generator=rng))
        retv[row, col] = 1.0

    return retv.tocsr()


def random_sparse_matrix(
    batch_size: int,
    num_rows: int,
    num_cols: int,
    fill: float,
    min_entries_per_col: int,
    rng: torch.Generator,
    device: DeviceType,
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
