# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
from typing import Any, Dict, List, Optional, Type, Union

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization, SparseLinearization
from theseus.optimizer.autograd import LUCudaSolveFunction

from .linear_solver import LinearSolver
from .utils import convert_to_alpha_beta_damping_tensors


class LUCudaSparseSolver(LinearSolver):
    # Class for keeping track of the inputs used when `reset()`` is called
    # Mostly useful for the `fill_defaults()` method
    @dataclasses.dataclass
    class ResetCtx:
        _DEFAULTS = {"batch_size": 16, "num_solver_contexts": 1}

        batch_size: Optional[int]
        num_solver_contexts: Optional[int]

        def fill_defaults(
            self, another_ctx: "LUCudaSparseSolver.ResetCtx"
        ) -> "LUCudaSparseSolver.ResetCtx":
            my_fields = dataclasses.asdict(self)
            other_fields = dataclasses.asdict(another_ctx)
            new_ctx_fields = copy.copy(LUCudaSparseSolver.ResetCtx._DEFAULTS)
            for k, v in my_fields.items():
                if v is None:
                    other_v = other_fields[k]
                    if other_v is not None:
                        new_ctx_fields[k] = other_v
                else:
                    new_ctx_fields[k] = v
            return LUCudaSparseSolver.ResetCtx(**new_ctx_fields)

    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        num_solver_contexts: Optional[int] = None,
        batch_size: Optional[int] = None,
        auto_reset: bool = True,
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

        self._last_reset_ctx = LUCudaSparseSolver.ResetCtx(None, None)
        if self.linearization.structure().num_rows:
            self.reset(batch_size=batch_size, num_solver_contexts=num_solver_contexts)

        self._objective = objective
        self._auto_reset = auto_reset

    def reset(
        self,
        batch_size: Optional[int] = None,
        num_solver_contexts: Optional[int] = None,
        **kwargs,
    ):
        # For any inputs that are None, this tries to set their values to
        # that used in the last call to `reset()`. If that value is also None,
        # (i.e., reset has never been called before) then it fills them
        # with the base default values
        ctx = LUCudaSparseSolver.ResetCtx(
            batch_size, num_solver_contexts
        ).fill_defaults(self._last_reset_ctx)
        # As a consequence of the above, reset() is only run if it either has
        # never been run before, or if at least one of the parameters is
        # explicitly requested to be different from those used in the last reset
        if ctx == self._last_reset_ctx:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Cuda not available, LUCudaSparseSolver cannot be used")
        try:
            from theseus.extlib.cusolver_lu_solver import CusolverLUSolver
        except Exception as e:
            raise RuntimeError(
                "Theseus C++/Cuda extension cannot be loaded\n"
                "even if Cuda appears to be available. Make sure Theseus\n"
                "is installed with Cuda support (export CUDA_HOME=...)\n"
                f"{type(e).__name__}: {e}"
            )

        self.A_row_ptr = torch.tensor(
            self.linearization.structure().row_ptr, dtype=torch.int32
        ).cuda()
        self.A_col_ind = torch.tensor(
            self.linearization.structure().col_ind, dtype=torch.int32
        ).cuda()
        At_mock = self.linearization.structure().mock_csc_transpose()
        AtA_mock = (At_mock @ At_mock.T).tocsr()

        # symbolic decomposition depending on the sparse structure, done with mock data
        # HACK: we generate several context, as by cublas the symbolic_decomposition is
        # also a context for factorization, and the two cannot be separated
        AtA_row_ptr = torch.tensor(AtA_mock.indptr, dtype=torch.int32).cuda()
        AtA_col_ind = torch.tensor(AtA_mock.indices, dtype=torch.int32).cuda()
        self._solver_contexts: List[CusolverLUSolver] = [
            CusolverLUSolver(
                ctx.batch_size,
                AtA_mock.shape[1],
                AtA_row_ptr,
                AtA_col_ind,
            )
            for _ in range(ctx.num_solver_contexts)
        ]
        self._last_solver_context: int = ctx.num_solver_contexts - 1
        self._last_reset_ctx = ctx

    def solve(
        self,
        damping: Optional[Union[float, torch.Tensor]] = None,
        ellipsoidal_damping: bool = True,
        damping_eps: float = 1e-8,
        **kwargs,
    ) -> torch.Tensor:
        if self._auto_reset:
            if self._solver_contexts[0].batch_size != self._objective.batch_size:
                self.reset(self._objective.batch_size)
        if not isinstance(self.linearization, SparseLinearization):
            raise RuntimeError(
                "LUCudaSparseSolver only works with theseus.optimizer.SparseLinearization."
            )

        self._last_solver_context = (
            self._last_solver_context + 1
        ) % self._last_reset_ctx.num_solver_contexts

        if damping is None:
            damping_alpha_beta = None
        else:
            damping_alpha_beta = convert_to_alpha_beta_damping_tensors(
                damping,
                damping_eps,
                ellipsoidal_damping,
                batch_size=self.linearization.A_val.shape[0],
                device=self.linearization.A_val.device,
                dtype=self.linearization.A_val.dtype,
            )

        return LUCudaSolveFunction.apply(
            self.linearization.A_val,
            self.linearization.b,
            self.linearization.structure(),
            self.A_row_ptr,
            self.A_col_ind,
            self._solver_contexts[self._last_solver_context],
            damping_alpha_beta,
            True,
            self.linearization.detached_hessian,
        )
