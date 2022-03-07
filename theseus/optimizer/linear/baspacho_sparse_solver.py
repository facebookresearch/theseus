# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization, SparseLinearization
from theseus.optimizer.autograd import LUCudaSolveFunction

from .linear_solver import LinearSolver


class LUCudaSparseSolver(LinearSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        num_solver_contexts=1,
        **kwargs,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda not available, LUCudaSparseSolver cannot be used")

        linearization_cls = linearization_cls or SparseLinearization
        if not linearization_cls == SparseLinearization:
            raise RuntimeError(
                "LUCudaSparseSolver only works with theseus.optimizer.SparseLinearization,"
                + f" got {type(self.linearization)}"
            )

        super().__init__(objective, linearization_cls, linearization_kwargs, **kwargs)
        self.linearization: SparseLinearization = self.linearization

        self._num_solver_contexts: int = num_solver_contexts

        if self.linearization.structure().num_rows:
            self.reset()

    def reset(self, batch_size: int = 16):
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda not available, LUCudaSparseSolver cannot be used")

        try:
            from theseus.extlib.baspacho_solver import CusolverLUSolver
        except Exception as e:
            raise RuntimeError( # FIXME
                "Theseus C++/Cuda extension cannot be loaded\n"
                "even if Cuda appears to be available. Make sure Theseus\n"
                "is installed with Cuda support (export CUDA_HOME=...)\n"
                f"{type(e).__name__}: {e}"
            )

        # convert to tensors for accelerated Mt x M operation
        # TODO: ".cuda" only if necessary
        self.A_rowPtr = torch.tensor(
            self.linearization.structure().row_ptr, dtype=torch.int64
        ) #.cuda()
        self.A_colInd = torch.tensor(
            self.linearization.structure().col_ind, dtype=torch.int64
        ) #.cuda()

        # compute block-structure of AtA. To do so we multiply the Jacobian's
        # transpose At by a matrix that collapses the rows to block-rows, ie
        # if i-th block starts at b_i, then a non-zero in b_i-th row becomes
        # a non-zero in i-th row.
        At_mock = self.linearization.structure().mock_csc_transpose()
        num_vars = len(self.linearization.var_start_cols)
        to_blocks = csr_matrix((np.ones(num_vars), 
                                self.linearization.var_start_cols,
                                np.arange(num_vars+1)),
                                (num_vars, self.linearization.num_cols))
        block_At_mock = to_blocks @ At_mock
        block_AtA_mock = (block_At_mock @ block_At_mock.T).tocsr()

        param_size = torch.tensor(self.linearization.var_dims, dtype=torch.int64)
        block_struct_ptrs = torch.tensor(block_AtA_mock.indptr, dtype=torch.int64)
        block_struct_inds = torch.tensor(block_AtA_mock.indices, dtype=torch.int64)
        self.symbolic_decomposition = BaspachoSymbolicDecomposition(
            param_size,
            block_struct_ptrs,
            block_struct_inds
        )

    def solve(
        self,
        damping: Optional[float] = None,
        ellipsoidal_damping: bool = True,
        damping_eps: float = 1e-8,
        **kwargs,
    ) -> torch.Tensor:
        if not isinstance(self.linearization, SparseLinearization):
            raise RuntimeError(
                "CholmodSparseSolver only works with theseus.optimizer.SparseLinearization."
            )

        if damping is None:
            damping_alpha_beta = None
        else:
            # See Nocedal and Wright, Numerical Optimization, pp. 260 and 261
            # https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
            damping_alpha_beta = (
                (damping, damping_eps) if ellipsoidal_damping else (0.0, damping)
            )

        return BaspachoSolveFunction.apply(
            self.linearization.A_val,
            self.linearization.b,
            self.linearization.structure(),
            self.A_rowPtr,
            self.A_colInd,
            self.symbolic_decomposition,
            damping_alpha_beta
        )
