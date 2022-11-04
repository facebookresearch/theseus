# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.linalg

from theseus.core import Objective
from theseus.optimizer import DenseLinearization, Linearization

from .linear_solver import LinearSolver


class DenseSolver(LinearSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        check_singular: bool = False,
    ):
        linearization_cls = linearization_cls or DenseLinearization
        if linearization_cls != DenseLinearization:
            raise RuntimeError(
                "DenseSolver only works with theseus.nonlinear.DenseLinearization, "
                f"but {linearization_cls} was provided."
            )
        super().__init__(objective, linearization_cls, linearization_kwargs)
        self.linearization: DenseLinearization = self.linearization
        self._check_singular = check_singular

    @staticmethod
    def _apply_damping(
        matrix: torch.Tensor,
        damping: Union[float, torch.Tensor],
        ellipsoidal: bool = True,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if matrix.ndim != 3:
            raise ValueError(
                "Matrix must have a 3 dimensions, the first one being a batch dimension."
            )
        _, n, m = matrix.shape
        if n != m:
            raise ValueError("Matrix must be square.")

        # See Nocedal and Wright, Numerical Optimization, pp. 260 and 261
        # https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
        damping = torch.as_tensor(damping).to(device=matrix.device, dtype=matrix.dtype)
        if ellipsoidal:
            damping = damping.view(-1, 1)
            # Add eps to guard against ill-conditioned matrix
            damped_D = torch.diag_embed(damping * matrix.diagonal(dim1=1, dim2=2) + eps)
        else:
            damping = damping.view(-1, 1, 1)
            damped_D = damping * torch.eye(
                n, device=matrix.device, dtype=matrix.dtype
            ).unsqueeze(0)
        return matrix + damped_D

    def _apply_damping_and_solve(
        self,
        Atb: torch.Tensor,
        AtA: torch.Tensor,
        damping: Optional[Union[float, torch.Tensor]] = None,
        ellipsoidal_damping: bool = True,
        damping_eps: float = 1e-8,
    ) -> torch.Tensor:
        if damping is not None:
            AtA = DenseSolver._apply_damping(
                AtA, damping, ellipsoidal=ellipsoidal_damping, eps=damping_eps
            )
        return self._solve_sytem(Atb, AtA)

    @abc.abstractmethod
    def _solve_sytem(self, Atb: torch.Tensor, AtA: torch.Tensor) -> torch.Tensor:
        pass

    def solve(
        self,
        damping: Optional[Union[float, torch.Tensor]] = None,
        ellipsoidal_damping: bool = True,
        damping_eps: float = 1e-8,
        **kwargs,
    ) -> torch.Tensor:
        if self._check_singular:
            AtA = self.linearization.AtA
            Atb = self.linearization.Atb
            with torch.no_grad():
                output = torch.zeros(AtA.shape[0], AtA.shape[1]).to(AtA.device)
                _, _, infos = torch.lu(AtA, get_infos=True)
                good_idx = infos.bool().logical_not()
                if not good_idx.all():
                    warnings.warn(
                        "Singular matrix found in batch, solution will be set "
                        "to all 0 for all singular matrices.",
                        RuntimeWarning,
                    )
            AtA = AtA[good_idx]
            Atb = Atb[good_idx]
            solution = self._apply_damping_and_solve(
                Atb,
                AtA,
                damping=damping,
                ellipsoidal_damping=ellipsoidal_damping,
                damping_eps=damping_eps,
            )
            output[good_idx] = solution
            return output
        else:
            return self._apply_damping_and_solve(
                self.linearization.Atb,
                self.linearization.AtA,
                damping=damping,
                ellipsoidal_damping=ellipsoidal_damping,
                damping_eps=damping_eps,
            )


class LUDenseSolver(DenseSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = DenseLinearization,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        check_singular: bool = False,
    ):
        super().__init__(
            objective,
            linearization_cls,
            linearization_kwargs,
            check_singular=check_singular,
        )

    def _solve_sytem(self, Atb: torch.Tensor, AtA: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(AtA, Atb).squeeze(2)


class CholeskyDenseSolver(DenseSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = DenseLinearization,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        check_singular: bool = False,
    ):
        super().__init__(
            objective,
            linearization_cls,
            linearization_kwargs,
            check_singular=check_singular,
        )

    def _solve_sytem(self, Atb: torch.Tensor, AtA: torch.Tensor) -> torch.Tensor:
        lower = torch.linalg.cholesky(AtA)
        return torch.cholesky_solve(Atb, lower).squeeze(2)
