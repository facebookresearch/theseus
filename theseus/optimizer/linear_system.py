# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix


class SparseStructure(abc.ABC):
    def __init__(
        self,
        col_ind: np.ndarray,
        row_ptr: np.ndarray,
        num_rows: int,
        num_cols: int,
        dtype: np.dtype = np.float_,  # type: ignore
    ):
        self.col_ind = col_ind
        self.row_ptr = row_ptr
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = dtype

    def csr_straight(self, val: torch.Tensor) -> csr_matrix:
        return csr_matrix(
            (val, self.col_ind, self.row_ptr),
            (self.num_rows, self.num_cols),
            dtype=self.dtype,
        )

    def csc_transpose(self, val: torch.Tensor) -> csc_matrix:
        return csc_matrix(
            (val, self.col_ind, self.row_ptr),
            (self.num_cols, self.num_rows),
            dtype=self.dtype,
        )

    def mock_csc_transpose(self) -> csc_matrix:
        return csc_matrix(
            (np.ones(len(self.col_ind), dtype=self.dtype), self.col_ind, self.row_ptr),
            (self.num_cols, self.num_rows),
            dtype=self.dtype,
        )
