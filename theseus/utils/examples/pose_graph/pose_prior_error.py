from typing import List, Optional, Tuple

import torch

import theseus as th


class PosePriorError(th.CostFunction):
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
        return PosePriorError(
            pose=self.pose,
            pose_prior=self.pose_prior,
            weight=self.weight,
            name=self.name,
        )
