# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

import theseus as th


class PosePirorError(th.CostFunction):
    def __init__(
        self,
        pose: th.SE3,
        pose_prior: th.SE3,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1e-5).to(dtype=pose.dtype))
        super().__init__(
            cost_weight=weight,
            name=name,
        )

        self.pose = pose
        self.pose_prior = pose_prior

        self.register_optim_vars(["pose"])
        self.register_aux_vars(["pose_prior"])

    def error(self) -> torch.Tensor:
        pose_err = self.pose_prior.between(self.pose)
        err = pose_err.log_map()
        return err

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pose_err = self.pose_prior.between(self.pose)
        err = pose_err.log_map()
        log_jac: List[torch.Tensor] = []
        err = pose_err.log_map(jacobians=log_jac)
        dlog = log_jac[0]

        return [dlog], err

    def dim(self) -> int:
        return 6

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self):
        return PosePirorError(
            pose=self.pose,
            pose_prior=self.pose_prior,
            weight=self.weight,
            name=self.name,
        )
