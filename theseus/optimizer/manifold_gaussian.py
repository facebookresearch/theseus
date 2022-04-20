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
            precision = torch.eye(self.dof).to(
                dtype=mean[0].dtype, device=mean[0].device
            )
            precision = precision[None, ...].repeat(mean[0].shape[0], 1, 1)
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
        precision_copy = self.precision.clone()
        return ManifoldGaussian(mean_copy, precision=precision_copy, name=new_name)

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

        expected_shape = torch.Size([mean[0].shape[0], self.dof, self.dof])
        if precision.shape != expected_shape:
            raise ValueError(
                f"Tried to update precision with data "
                f"incompatible with original tensor shape. Given: {precision.shape}. "
                f"Expected: {expected_shape}"
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


# Projects the gaussian (ManifoldGaussian object) into the tangent plane at
# variable. The gaussian mean is projected using the local function,
# and the precision is approximately transformed using the jacobains of the exp_map.
# Either returns the mean and precision of the new Gaussian in the tangent plane if
# return_mean is True. Otherwise returns the information vector (eta) and precision.
# See section H, eqn 55 in https://arxiv.org/pdf/1812.01537.pdf for a derivation
# of covariance propagation in manifolds.
def local_gaussian(
    variable: LieGroup,
    gaussian: ManifoldGaussian,
    return_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # assumes gaussian is over just one Manifold object
    if len(gaussian.mean) != 1:
        raise ValueError(
            "local on ManifoldGaussian should be over just one Manifold object. "
            f"Passed gaussian {gaussian.name} is over {len(gaussian.mean)} "
            "Manifold objects."
        )
    # check variable and gaussian are of the same LieGroup class
    if gaussian.mean[0].__class__ != variable.__class__:
        raise ValueError(
            "variable and gaussian mean must be instances of the same class. "
            f"variable is of class {variable.__class__} and gaussian mean is "
            f"of class {gaussian.mean[0].__class__}."
        )

    # mean vector in the tangent space at variable
    mean_tp = variable.local(gaussian.mean[0])

    jac: List[torch.Tensor] = []
    variable.exp_map(mean_tp, jacobians=jac)
    # precision matrix in the tangent space at variable
    lam_tp = torch.bmm(torch.bmm(jac[0].transpose(-1, -2), gaussian.precision), jac[0])

    if return_mean:
        return mean_tp, lam_tp
    else:
        eta_tp = torch.matmul(lam_tp, mean_tp.unsqueeze(-1)).squeeze(-1)
        return eta_tp, lam_tp


# Computes the ManifoldGaussian that corresponds to the gaussian in the tangent plane
# at variable, parameterised by the mean (mean_tp) and precision (precision_tp).
# The mean is transformed to a LieGroup element by retraction.
# The precision is transformed using the inverse of the exp_map jacobians.
# See section H, eqn 55 in https://arxiv.org/pdf/1812.01537.pdf for a derivation
# of covariance propagation in manifolds.
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
