from typing import List, Optional, Tuple, Union

import torch

import theseus as th


class RelativePoseError(th.CostFunction):
    def __init__(
        self,
        pose1: Union[th.SE2, th.SE3],
        pose2: Union[th.SE2, th.SE3],
        relative_pose: Union[th.SE2, th.SE3],
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=pose1.dtype))
        super().__init__(
            cost_weight=weight,
            name=name,
        )

        self.pose1 = pose1
        self.pose2 = pose2
        self.relative_pose = relative_pose

        self.register_optim_vars(["pose1", "pose2"])
        self.register_aux_vars(["relative_pose"])

    def error(self) -> torch.Tensor:
        pose_comp = th.SE3.compose(self.pose1, self.relative_pose)
        pose_err = self.pose2.between(pose_comp)
        err = pose_err.log_map()
        return err

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        pose_comp = th.SE3.compose(self.pose1, self.relative_pose)
        pose_err = self.pose2.between(pose_comp)
        log_jac: List[torch.Tensor] = []
        err = pose_err.log_map(jacobians=log_jac)
        dlog = log_jac[0]

        return [
            dlog @ self.relative_pose.inverse().adjoint(),
            -dlog @ pose_err.inverse().adjoint(),
        ], err

    def dim(self) -> int:
        return 6

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self):
        return RelativePoseError(
            pose1=self.pose1,
            pose2=self.pose2,
            relative_pose=self.relative_pose,
            weight=self.weight,
            name=self.name,
        )
