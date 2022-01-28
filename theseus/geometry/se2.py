# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

import theseus.constants

from .lie_group import LieGroup
from .point_types import Point2
from .so2 import SO2


# If data is passed, must be x, y, cos, sin
# If x_y_theta is passed, must be tensor with shape batch_size x 3
class SE2(LieGroup):
    SE2_EPS = 5e-7

    def __init__(
        self,
        x_y_theta: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if x_y_theta is not None and data is not None:
            raise ValueError("Please provide only one of x_y_theta or data.")
        if x_y_theta is not None:
            dtype = x_y_theta.dtype
        super().__init__(data=data, name=name, dtype=dtype)
        if x_y_theta is not None:
            rotation = SO2(theta=x_y_theta[:, 2:])
            translation = Point2(data=x_y_theta[:, :2])
            self.update_from_rot_and_trans(rotation, translation)

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.tensor([0.0, 0.0, 1.0, 0.0]).view(1, 4)

    def dof(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"SE2(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            theta = torch.atan2(self[:, 3:], self[:, 2:3])
            xytheta = torch.cat([self[:, :2], theta], dim=1)
            return f"SE2(xytheta={xytheta}, name={self.name})"

    @property
    def rotation(self) -> SO2:
        return SO2(data=self[:, 2:])

    def theta(self, jacobians: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            J_out = torch.zeros(
                self.shape[0], 1, 3, device=self.device, dtype=self.dtype
            )
            J_out[:, 0, 2] = 1
            jacobians.append(J_out)
        return self.rotation.theta()

    @property
    def translation(self) -> Point2:
        return Point2(data=self[:, :2])

    def xy(self, jacobians: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            rotation = self.rotation
            J_out = torch.zeros(
                self.shape[0], 2, 3, device=self.device, dtype=self.dtype
            )
            J_out[:, :2, :2] = rotation.to_matrix()
            jacobians.append(J_out)
        return self.translation.data

    def update_from_rot_and_trans(self, rotation: SO2, translation: Point2):
        batch_size = rotation.shape[0]
        self.data = torch.empty(batch_size, 4).to(
            device=rotation.device, dtype=rotation.dtype
        )
        self[:, :2] = translation.data
        cosine, sine = rotation.to_cos_sin()
        self[:, 2] = cosine
        self[:, 3] = sine

    # From https://github.com/strasdat/Sophus/blob/master/sophus/se2.hpp#L160
    def _log_map_impl(self) -> torch.Tensor:
        rotation = self.rotation
        theta = rotation.log_map().view(-1)
        cth, sth = rotation.to_cos_sin()
        small_theta = theta.abs() < 1e-7

        # Computer the approximations when theta is near to 0
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)
        a = (
            0.5
            * (1 + cth)
            * torch.where(
                small_theta,
                1 + sth ** 2 / 6,
                theta / torch.where(small_theta, non_zero, sth),
            )
        )
        b = 0.5 * theta

        return torch.stack(
            (a * self[:, 0] + b * self[:, 1], a * self[:, 1] - b * self[:, 0], theta),
            dim=1,
        )

    # From https://github.com/strasdat/Sophus/blob/master/sophus/se2.hpp#L558
    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        u = tangent_vector[:, :2]
        theta = tangent_vector[:, 2]
        rotation = SO2(theta=theta)

        cosine, sine = rotation.to_cos_sin()

        sin_theta_by_theta = torch.zeros_like(sine)
        one_minus_cos_theta_by_theta = torch.zeros_like(sine)

        # Compute above quantities when theta is not near zero
        idx_regular_thetas = theta.abs() > theseus.constants.EPS
        if idx_regular_thetas.any():
            sin_theta_by_theta[idx_regular_thetas] = (
                sine[idx_regular_thetas] / theta[idx_regular_thetas]
            )
            one_minus_cos_theta_by_theta[idx_regular_thetas] = (
                -cosine[idx_regular_thetas] + 1
            ) / theta[idx_regular_thetas]

        # Same as above three lines but for small angles
        idx_small_thetas = theta.abs() < theseus.constants.EPS
        if idx_small_thetas.any():
            small_theta = theta[idx_small_thetas]
            small_theta_sq = small_theta ** 2
            sin_theta_by_theta[idx_small_thetas] = -small_theta_sq / 6 + 1
            one_minus_cos_theta_by_theta[idx_small_thetas] = (
                0.5 * small_theta - small_theta / 24 * small_theta_sq
            )

        new_x = sin_theta_by_theta * u[:, 0] - one_minus_cos_theta_by_theta * u[:, 1]
        new_y = one_minus_cos_theta_by_theta * u[:, 0] + sin_theta_by_theta * u[:, 1]
        translation = Point2(data=torch.stack([new_x, new_y], dim=1))

        se2 = SE2(dtype=tangent_vector.dtype)
        se2.update_from_rot_and_trans(rotation, translation)
        return se2

    def _adjoint_impl(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 3, 3).to(device=self.device, dtype=self.dtype)
        ret[:, :2, :2] = self.rotation.to_matrix()
        ret[:, 0, 2] = self[:, 1]
        ret[:, 1, 2] = -self[:, 0]
        ret[:, 2, 0] = 0
        ret[:, 2, 2] = 1
        return ret

    def _compose_impl(self, se2_2: LieGroup) -> "SE2":
        se2_2 = cast(SE2, se2_2)
        rotation_1 = self.rotation
        rotation_2 = se2_2.rotation
        translation_1 = cast(Point2, self.translation)
        translation_2 = cast(Point2, se2_2.translation)
        new_rotation = rotation_1.compose(rotation_2)
        new_translation = cast(
            Point2,
            translation_1.compose(rotation_1.rotate(translation_2)),
        )
        return SE2(data=torch.cat([new_translation.data, new_rotation.data], dim=1))

    def _inverse_impl(self) -> "SE2":
        inverse_rotation = self.rotation._inverse_impl()
        inverse_translation = inverse_rotation.rotate(cast(Point2, -self.translation))
        se2_inverse = SE2(dtype=self.dtype)
        se2_inverse.update_from_rot_and_trans(inverse_rotation, inverse_translation)
        return se2_inverse

    def to_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(self.shape[0], 3, 3).to(
            device=self.device, dtype=self.dtype
        )
        matrix[:, :2, :2] = self.rotation.to_matrix()
        matrix[:, :2, 2] = self[:, :2]
        matrix[:, 2, 2] = 1.0
        return matrix

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        theta = tangent_vector[:, 2]
        u = tangent_vector[:, :2]
        matrix = torch.zeros(tangent_vector.shape[0], 3, 3).to(
            device=tangent_vector.device, dtype=tangent_vector.dtype
        )
        matrix[:, 0, 1] = -theta
        matrix[:, 1, 0] = theta
        matrix[:, :2, 2] = u
        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        _check = matrix.ndim == 3 and matrix.shape[1:] == (3, 3)
        _check &= torch.allclose(matrix[:, 0, 1], -matrix[:, 1, 0])
        _check &= matrix[:, 0, 0].abs().max().item() < theseus.constants.EPS
        _check &= matrix[:, 1, 1].abs().max().item() < theseus.constants.EPS
        _check &= matrix[:, 2, 2].abs().max().item() < theseus.constants.EPS
        if not _check:
            raise ValueError("Invalid hat matrix for SE2.")
        batch_size = matrix.shape[0]
        tangent_vector = torch.zeros(batch_size, 3).to(
            device=matrix.device, dtype=matrix.dtype
        )
        tangent_vector[:, 2] = matrix[:, 1, 0]
        tangent_vector[:, :2] = matrix[:, :2, 2]
        return tangent_vector

    def transform_to(
        self,
        point: Union[torch.Tensor, Point2],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point2:
        point_data = point if isinstance(point, torch.Tensor) else point.data
        translation_rel = point_data - self.xy()
        J_rot: Optional[List[torch.Tensor]] = None
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            J_rot = []
        transform = self.rotation.unrotate(translation_rel, jacobians=J_rot)
        if jacobians is not None:
            J_rot_pose, J_rot_point = J_rot
            J_out_pose = torch.zeros(
                self.shape[0], 2, 3, device=self.device, dtype=self.dtype
            )
            J_out_pose[:, :2, :2] = -torch.eye(
                2, device=self.device, dtype=self.dtype
            ).unsqueeze(0)
            J_out_pose[:, :, 2:] = J_rot_pose
            jacobians.extend([J_out_pose, J_rot_point])
        return transform

    def _copy_impl(self, new_name: Optional[str] = None) -> "SE2":
        return SE2(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SE2":
        return cast(SE2, super().copy(new_name=new_name))
