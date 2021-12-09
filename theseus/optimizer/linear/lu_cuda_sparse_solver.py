from typing import Any, Dict, List, Optional, Type

import torch

from theseus.core import Objective
from theseus.extlib.cusolver_lu_solver import CusolverLUSolver
from theseus.optimizer import Linearization, SparseLinearization
from theseus.optimizer.autograd import LUCudaSolveFunction

from .linear_solver import LinearSolver


class LUCudaSparseSolver(LinearSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        damping: float = 1e-6,
        batch_size: int = 16,
        num_solver_contexts=1,
        **kwargs,
    ):
        linearization_cls = linearization_cls or SparseLinearization
        if not linearization_cls == SparseLinearization:
            raise RuntimeError(
                "LUCudaSparseSolver only works with theseus.optimizer.SparseLinearization,"
                + f" got {type(self.linearization)}"
            )

        super().__init__(objective, linearization_cls, linearization_kwargs, **kwargs)
        self.linearization: SparseLinearization = self.linearization

        if self.linearization.structure().num_rows:
            self.reset()

        # the `damping` has the purpose of (optionally) improving conditioning
        # self._damping: float = damping

        self._num_solver_contexts: int = num_solver_contexts

    def reset(self, batch_size: int = 16):
        self.A_rowPtr = torch.tensor(
            self.linearization.structure().row_ptr, dtype=torch.int32
        ).cuda()
        self.A_colInd = torch.tensor(
            self.linearization.structure().col_ind, dtype=torch.int32
        ).cuda()
        At_mock = self.linearization.structure().mock_csc_transpose()
        AtA_mock = (At_mock @ At_mock.T).tocsr()

        # symbolic decomposition depending on the sparse structure, done with mock data
        # HACK: we generate several context, as by cublas the symbolic_decomposition is
        # also a context for factorization, and the two cannot be separated
        AtA_rowPtr = torch.tensor(AtA_mock.indptr, dtype=torch.int32).cuda()
        AtA_colInd = torch.tensor(AtA_mock.indices, dtype=torch.int32).cuda()
        self._solver_contexts: List[CusolverLUSolver] = [
            CusolverLUSolver(
                batch_size,
                AtA_mock.shape[1],
                AtA_rowPtr,
                AtA_colInd,
            )
            for _ in range(self._num_solver_contexts)
        ]
        self._last_solver_context: int = self._num_solver_contexts - 1

    def solve(self, damping: Optional[float] = None, **kwargs) -> torch.Tensor:
        if not isinstance(self.linearization, SparseLinearization):
            raise RuntimeError(
                "CholmodSparseSolver only works with theseus.optimizer.SparseLinearization."
            )

        self._last_solver_context = (
            self._last_solver_context + 1
        ) % self._num_solver_contexts

        return LUCudaSolveFunction.apply(
            self.linearization.A_val,
            self.linearization.b,
            self.linearization.structure(),
            self.A_rowPtr,
            self.A_colInd,
            self._solver_contexts[self._last_solver_context],
            True,
            # self._damping,
        )
