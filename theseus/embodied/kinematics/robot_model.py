# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch

from theseus.geometry import SE2, LieGroup, Point2, Vector


class RobotModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward_kinematics(self, robot_pose: LieGroup) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def dim(self) -> int:
        pass


class IdentityModel(RobotModel):
    def __init__(self):
        super().__init__()

    def forward_kinematics(self, robot_pose: LieGroup) -> torch.Tensor:
        if isinstance(robot_pose, SE2):
            return robot_pose.translation.data.view(-1, 2, 1)
        if isinstance(robot_pose, Point2) or isinstance(robot_pose, Vector):
            assert robot_pose.dof() == 2
            return robot_pose.data.view(-1, 2, 1)
        raise NotImplementedError(
            f"IdentityModel not implemented for pose with type {type(robot_pose)}."
        )

    def dim(self) -> int:
        return 1
