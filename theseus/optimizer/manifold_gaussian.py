# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import count
from typing import List, Optional, Sequence, Tuple

import torch

from theseus.geometry import LieGroup, Manifold


class ManifoldGaussian:
    _ids = count(0)

    def __init__(
        self,
        mean: Sequence[Manifold],
        precision: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ):
        self._id = next(ManifoldGaussian._ids)
        if name is None:
            name = f"{self.__class__.__name__}__{self._id}"
        self.name = name

        dof = 0
        for v in mean:
            dof += v.dof()
        self._dof = dof

        self.mean = mean
        if precision is None:
            precision = torch.zeros(mean[0].shape[0], self.dof, self.dof).to(
                dtype=mean[0].dtype, device=mean[0].device
            )
        self.update(mean, precision)

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def device(self) -> torch.device:
        return self.mean[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.mean[0].dtype

    # calls to() on the internal tensors
    def to(self, *args, **kwargs):
        for var in self.mean:
            var = var.to(*args, **kwargs)
        self.precision = self.precision.to(*args, **kwargs)

    def copy(self, new_name: Optional[str] = None) -> "ManifoldGaussian":
        if not new_name:
            new_name = f"{self.name}_copy"
        mean_copy = [var.copy() for var in self.mean]
        return ManifoldGaussian(mean_copy, name=new_name)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        the_copy = self.copy()
        memo[id(self)] = the_copy
        return the_copy

    def update(
        self,
        mean: Sequence[Manifold],
        precision: torch.Tensor,
    ):
        if len(mean) != len(self.mean):
            raise ValueError(
                f"Tried to update mean with sequence of different"
                f"length to original mean sequence. Given: {len(mean)}. "
                f"Expected: {len(self.mean)}"
            )
        for i in range(len(self.mean)):
            self.mean[i].update(mean[i])

        if precision.shape != torch.Size([mean[0].shape[0], self.dof, self.dof]):
            raise ValueError(
                f"Tried to update precision with data "
                f"incompatible with original tensor shape. Given: {precision.shape}. "
                f"Expected: {self.precision.shape}"
            )
        if precision.dtype != self.dtype:
            raise ValueError(
                f"Tried to update using tensor of dtype: {precision.dtype} but precision "
                f"has dtype: {self.dtype}."
            )
        if precision.device != self.device:
            raise ValueError(
                f"Tried to update using tensor on device: {precision.dtype} but precision "
                f"is on device: {self.device}."
            )
        if not torch.allclose(precision, precision.transpose(1, 2)):
            raise ValueError("Tried to update precision with non-symmetric matrix.")

        self.precision = precision


def local_gaussian(
    variable: LieGroup,
    gaussian: ManifoldGaussian,
    return_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # mean vector in the tangent space at variable
    mean_tp = variable.local(gaussian.mean[0])

    jac: List[torch.Tensor] = []
    variable.exp_map(mean_tp, jacobians=jac)
    # precision matrix in the tangent space at variable
    # Following math in section H https://arxiv.org/pdf/1812.01537.pdf
    lam_tp = torch.bmm(torch.bmm(jac[0].transpose(-1, -2), gaussian.precision), jac[0])

    if return_mean:
        return mean_tp, lam_tp
    else:
        eta_tp = torch.matmul(lam_tp, mean_tp.unsqueeze(-1)).squeeze(-1)
        return eta_tp, lam_tp


def retract_gaussian(
    variable: LieGroup,
    mean_tp: torch.Tensor,
    precision_tp: torch.Tensor,
) -> ManifoldGaussian:
    mean = variable.retract(mean_tp)

    jac: List[torch.Tensor] = []
    variable.exp_map(mean_tp, jacobians=jac)
    inv_jac = torch.inverse(jac[0])
    precision = torch.bmm(torch.bmm(inv_jac.transpose(-1, -2), precision_tp), inv_jac)

    return ManifoldGaussian(mean=[mean], precision=precision)
