# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Type, Union

import torch

# rename Cholesky `Factor` to CholeskyDecomposition to
# prevent confusion with the factors of the factor graph,
# when using the probabilistic naming convention
from sksparse.cholmod import Factor as CholeskyDecomposition
from sksparse.cholmod import analyze_AAt

from theseus.core import Objective
from theseus.optimizer import Linearization, SparseLinearization
from theseus.optimizer.autograd import CholmodSolveFunction

from .linear_solver import LinearSolver


class CholmodSparseSolver(LinearSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        damping: float = 1e-6,
        **kwargs,
    ):
        linearization_cls = linearization_cls or SparseLinearization
        if not linearization_cls == SparseLinearization:
            raise RuntimeError(
                "CholmodSparseSolver only works with theseus.optimizer.SparseLinearization,"
                + f" got {type(self.linearization)}"
            )

        super().__init__(objective, linearization_cls, linearization_kwargs, **kwargs)
        self.linearization: SparseLinearization = self.linearization

        # symbolic decomposition depending on the sparse structure, done with mock data
        self._symbolic_cholesky_decomposition: CholeskyDecomposition = analyze_AAt(
            self.linearization.structure().mock_csc_transpose()
        )

        # the `damping` has the purpose of (optionally) improving conditioning
        self._damping: float = damping

    def solve(
        self, damping: Optional[Union[float, torch.Tensor]] = None, **kwargs
    ) -> torch.Tensor:
        if damping is not None and not isinstance(damping, float):
            raise ValueError("CholmodSparseSolver only supports scalar damping.")
        damping = damping or self._damping
        if not isinstance(self.linearization, SparseLinearization):
            raise RuntimeError(
                "CholmodSparseSolver only works with theseus.optimizer.SparseLinearization."
            )

        return CholmodSolveFunction.apply(
            self.linearization.A_val,
            self.linearization.b,
            self.linearization.structure(),
            self._symbolic_cholesky_decomposition,
            damping,
        )
