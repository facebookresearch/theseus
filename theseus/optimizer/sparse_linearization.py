# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import numpy as np
import torch

from theseus.core import Objective
from theseus.utils.sparse_matrix_utils import sparse_mv, sparse_mtv

from .linear_system import SparseStructure
from .linearization import Linearization
from .variable_ordering import VariableOrdering


class SparseLinearization(Linearization):
    def __init__(
        self,
        objective: Objective,
        ordering: Optional[VariableOrdering] = None,
        **kwargs,
    ):
        super().__init__(objective, ordering)

        # we prepare the indices for At as csc matrix (or A as csr, same thing)
        # for similarity with dense_linearization code we build A as csr, then
        # actually we have At as csc and can feed it to `cholesky_AAt` routine
        # we want a unique set of rowPtr/colInd indices for A for all batches
        # we also save pointers to the data block, so that we can later quickly
        # write the data blocks
        A_col_ind: List[int] = []
        A_row_ptr: List[int] = [0]

        # ptr to data block (stride = sum of variable.dim()
        cost_function_block_pointers = []
        cost_function_row_block_starts = []  # where data start for this row block
        cost_function_stride = []  # total jacobian cols

        for _, cost_function in enumerate(self.objective._get_jacobians_iter()):
            num_rows = cost_function.dim()
            col_slices_indices = []
            for var_idx_in_cost_function, variable in enumerate(
                cost_function.optim_vars
            ):
                var_idx_in_order = self.ordering.index_of(
                    cost_function.optim_var_at(var_idx_in_cost_function).name
                )
                var_start_col = self.var_start_cols[var_idx_in_order]
                num_cols = variable.dof()
                col_slice = slice(var_start_col, var_start_col + num_cols)
                col_slices_indices.append((col_slice, var_idx_in_cost_function))

            # sort according to how they will be written inside A
            col_slices_indices.sort()
            sorted_block_sizes = [(s.stop - s.start) for s, _ in col_slices_indices]
            sorted_block_pointers = np.cumsum([0] + sorted_block_sizes)[:-1]
            sorted_indices = np.array([i for _, i in col_slices_indices])
            block_pointers: np.ndarray = np.ndarray(
                (len(col_slices_indices),), dtype=int
            )
            block_pointers[sorted_indices] = sorted_block_pointers
            cost_function_block_pointers.append(block_pointers)

            cost_function_row_block_starts.append(len(A_col_ind))
            col_ind = [c for s, _ in col_slices_indices for c in range(s.start, s.stop)]
            cost_function_stride.append(len(col_ind))

            for _ in range(num_rows):
                A_col_ind += col_ind
                A_row_ptr.append(len(A_col_ind))

        # not batched, these data are the same across batches
        self.cost_function_block_pointers = cost_function_block_pointers
        self.cost_function_row_block_starts: np.ndarray = np.array(
            cost_function_row_block_starts, dtype=int
        )
        self.cost_function_stride: np.ndarray = np.array(
            cost_function_stride, dtype=int
        )
        self.A_row_ptr: np.ndarray = np.array(A_row_ptr, dtype=int)
        self.A_col_ind: np.ndarray = np.array(A_col_ind, dtype=int)

        # batched data
        self.A_val: torch.Tensor = None
        self.b: torch.Tensor = None

        # computed lazily by self._atb_impl() and reset to None by
        # self._linearize_jacobian_impl()
        self._Atb: torch.Tensor = None

        # computed lazily by self.diagonal_scaling() and reset to None by
        # self._linearize_jacobian_impl()
        self._AtA_diag: torch.Tensor = None

        # If true, it signals to linear solvers that any computation resulting from
        # matrix (At * A) must be detached from the compute graph
        self.detached_hessian = False

    def _linearize_jacobian_impl(self):
        self._detached_hessian = False
        self._Atb = None
        self._AtA_diag = None

        # those will be fully overwritten, no need to zero:
        self.A_val = torch.empty(
            size=(self.objective.batch_size, len(self.A_col_ind)),
            device=self.objective.device,
            dtype=self.objective.dtype,
        )
        self.b = torch.empty(
            size=(self.objective.batch_size, self.num_rows),
            device=self.objective.device,
            dtype=self.objective.dtype,
        )

        err_row_idx = 0
        for f_idx, cost_function in enumerate(self.objective._get_jacobians_iter()):
            jacobians, error = cost_function.weighted_jacobians_error()
            num_rows = cost_function.dim()
            row_slice = slice(err_row_idx, err_row_idx + num_rows)

            # we will view the blocks of rows inside `A_val` as `num_rows` x `stride` matrix
            block_start = self.cost_function_row_block_starts[f_idx]
            stride = self.cost_function_stride[f_idx]
            block = self.A_val[:, block_start : block_start + stride * num_rows].view(
                -1, num_rows, stride
            )
            block_pointers = self.cost_function_block_pointers[f_idx]

            for var_idx_in_cost_function, var_jacobian in enumerate(jacobians):
                # the proper block is written, using the precomputed index in `block_pointers`
                num_cols = var_jacobian.shape[2]
                pointer = block_pointers[var_idx_in_cost_function]
                block[:, :, pointer : pointer + num_cols] = var_jacobian

            self.b[:, row_slice] = -error
            err_row_idx += cost_function.dim()

    def structure(self):
        return SparseStructure(
            self.A_col_ind,
            self.A_row_ptr,
            self.num_rows,
            self.num_cols,
            dtype=np.float64 if self.objective.dtype == torch.double else np.float32,
        )

    def _linearize_hessian_impl(self, _detach_hessian: bool = False):
        # Some of our sparse solvers don't require explicitly computing the
        # hessian approximation, so we only compute the jacobian here and let each
        # solver handle this as needed
        self._linearize_jacobian_impl()
        self.detached_hessian = _detach_hessian

    def _ata_impl(self) -> torch.Tensor:
        raise NotImplementedError("AtA is not yet implemented for SparseLinearization.")

    def _atb_impl(self) -> torch.Tensor:
        if self._Atb is None:
            A_row_ptr = torch.tensor(self.A_row_ptr, dtype=torch.int32).to(
                self.objective.device
            )
            A_col_ind = A_row_ptr.new_tensor(self.A_col_ind)

            # unsqueeze at the end for consistency with DenseLinearization
            self._Atb = sparse_mtv(
                self.num_cols,
                A_row_ptr,
                A_col_ind,
                self.A_val.double(),
                self.b.double(),
            ).unsqueeze(2)
        return self._Atb.to(dtype=self.A_val.dtype)

    def Av(self, v: torch.Tensor) -> torch.Tensor:
        A_row_ptr = torch.tensor(self.A_row_ptr, dtype=torch.int32).to(
            self.objective.device
        )
        A_col_ind = A_row_ptr.new_tensor(self.A_col_ind)
        return sparse_mv(
            self.num_cols, A_row_ptr, A_col_ind, self.A_val.double(), v.double()
        ).to(v.dtype)

    def diagonal_scaling(self, v: torch.Tensor) -> torch.Tensor:
        assert v.ndim == 2
        assert v.shape[1] == self.num_cols
        if self._AtA_diag is None:
            A_val = self.A_val
            self._AtA_diag = torch.zeros(A_val.shape[0], self.num_cols)
            for row in range(self.num_rows):
                start = self.A_row_ptr[row]
                end = self.A_row_ptr[row + 1]
                columns = self.A_col_ind[start:end]
                self._AtA_diag[:, columns] += A_val[:, start:end] ** 2
        return self._AtA_diag * v
