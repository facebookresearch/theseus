# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

import theseus
import theseus.constants

from .lie_group import LieGroup
from .point_types import Point3
from .so3 import SO3


class SE3(LieGroup):
    def __init__(
        self,
        x_y_z_quaternion: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if x_y_z_quaternion is not None and data is not None:
            raise ValueError("Please provide only one of x_y_z_quaternion or data.")
        if x_y_z_quaternion is not None:
            dtype = x_y_z_quaternion.dtype
        if data is not None:
            self._SE3_matrix_check(data)
        super().__init__(data=data, name=name, dtype=dtype)
        if x_y_z_quaternion is not None:
            self.update_from_x_y_z_quaternion(x_y_z_quaternion=x_y_z_quaternion)

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.eye(3, 4).view(1, 3, 4)

    def dof(self) -> int:
        return 6

    def __repr__(self) -> str:
        return f"SE3(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            return f"SE3(matrix={self.data}), name={self.name})"

    def _adjoint_impl(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 6, 6).to(dtype=self.dtype, device=self.device)
        ret[:, :3, :3] = self[:, :3, :3]
        ret[:, 3:, 3:] = self[:, :3, :3]
        ret[:, :3, 3:] = SO3.hat(self[:, :3, 3]) @ self[:, :3, :3]

        return ret

    @staticmethod
    def _SE3_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 4):
            raise ValueError("SE(3) can only be 3x4 matrices.")
        SO3._SO3_matrix_check(matrix.data[:, :3, :3])

    @staticmethod
    def x_y_z_unit_quaternion_to_SE3(x_y_z_quaternion: torch.Tensor) -> "SE3":
        if x_y_z_quaternion.ndim == 1:
            x_y_z_quaternion = x_y_z_quaternion.unsqueeze(0)

        if x_y_z_quaternion.ndim != 2 and x_y_z_quaternion.shape[1] != 7:
            raise ValueError("x_y_z_quaternion can only be 7-D vectors.")

        ret = SE3()

        batch_size = x_y_z_quaternion.shape[0]
        ret.data = torch.empty(batch_size, 3, 4).to(
            device=x_y_z_quaternion.device, dtype=x_y_z_quaternion.dtype
        )
        ret[:, :3, :3] = SO3.unit_quaternion_to_SO3(x_y_z_quaternion[:, 3:]).data
        ret[:, :3, 3] = x_y_z_quaternion[:, :3]

        return ret

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
            raise ValueError("Hat matrices of SE(3) can only be 4x4 matrices")

        if matrix[:, 3].abs().max().item() > theseus.constants.EPS:
            raise ValueError("The last row of hat matrices of SE(3) can only be zero.")

        if (
            matrix[:, :3, :3].transpose(1, 2) + matrix[:, :3, :3]
        ).abs().max().item() > theseus.constants.EPS:
            raise ValueError(
                "The 3x3 top-left corner of hat matrices of SE(3) can only be skew-symmetric."
            )

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 6:
            raise ValueError("Tangent vectors of SE(3) can only be 6-D vectors.")
        ret = SE3(dtype=tangent_vector.dtype)

        NEAR_ZERO_EPS = 5e-3
        tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
        tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

        theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
        theta2 = theta**2
        theta3 = theta**3

        near_zero = theta < NEAR_ZERO_EPS
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)
        theta3_nz = torch.where(near_zero, non_zero, theta3)

        # Compute the rotation
        sine = theta.sin()
        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine_by_theta = torch.where(
            near_zero, 0.5 * cosine + 0.5, theta.sin() / theta_nz
        )
        one_minus_cosie_by_theta2 = torch.where(
            near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
        )
        ret.data = torch.zeros(tangent_vector.shape[0], 3, 4).to(
            dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        ret.data[:, :3, :3] = (
            one_minus_cosie_by_theta2
            * tangent_vector_ang
            @ tangent_vector_ang.transpose(1, 2)
        )

        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
        ret[:, 0, 1] -= temp[:, 2]
        ret[:, 1, 0] += temp[:, 2]
        ret[:, 0, 2] += temp[:, 1]
        ret[:, 2, 0] -= temp[:, 1]
        ret[:, 1, 2] -= temp[:, 0]
        ret[:, 2, 1] += temp[:, 0]

        # Compute the translation
        sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
        one_minus_cosie_by_theta2 = torch.where(
            near_zero, 0.5 - theta2 / 24, one_minus_cosie_by_theta2
        )
        theta_minus_sine_by_theta3 = torch.where(
            near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
        )

        ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
        ret[:, :, 3:] += one_minus_cosie_by_theta2 * torch.cross(
            tangent_vector_ang, tangent_vector_lin
        )
        ret[:, :, 3:] += theta_minus_sine_by_theta3 * (
            tangent_vector_ang
            @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
        )

        return ret

    def _log_map_impl(self) -> torch.Tensor:
        raise NotImplementedError

    def _compose_impl(self, so3_2: LieGroup) -> "SE3":
        raise NotImplementedError

    def _inverse_impl(self, get_jacobian: bool = False) -> "SE3":
        ret = torch.zeros(self.shape[0], 3, 4).to(dtype=self.dtype, device=self.device)
        ret[:, :, :3] = self.data[:, :3, :3].transpose(1, 2)
        ret[:, :, 3] = -(ret[:, :3, :3] @ self.data[:, :3, 3].unsqueeze(2)).view(-1, 3)
        return SE3(data=ret)

    def to_matrix(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 4, 4).to(dtype=self.dtype, device=self.device)
        ret[:, :3] = self.data
        ret[:, 3, 3] = 1
        return ret

    def update_from_x_y_z_quaternion(self, x_y_z_quaternion: torch.Tensor):
        self.update(SE3.x_y_z_unit_quaternion_to_SE3(x_y_z_quaternion))

    def update_from_rot_and_trans(self, rotation: SO3, translation: Point3):
        if rotation.shape[0] != translation.shape[0]:
            raise ValueError("rotation and translation must have the same batch size.")

        if rotation.dtype != translation.dtype:
            raise ValueError("rotation and translation must be of the same type.")

        if rotation.device != translation.device:
            raise ValueError("rotation and translation must be on the same device.")

        self.data = torch.cat((rotation.data, translation.data.unsqueeze(2)), dim=2)

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        _check = tangent_vector.ndim == 2 and tangent_vector.shape[1] == 6
        if not _check:
            raise ValueError("Invalid vee matrix for SE(3).")
        matrix = torch.zeros(tangent_vector.shape[0], 4, 4).to(
            dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        matrix[:, :3, :3] = SO3.hat(tangent_vector[:, 3:])
        matrix[:, :3, 3] = tangent_vector[:, :3]

        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        SE3._hat_matrix_check(matrix)
        return torch.cat((matrix[:, :3, 3], SO3.vee(matrix[:, :3, :3])), dim=1)

    def _transform_shape_check(self, point: Union[Point3, torch.Tensor]):
        raise NotImplementedError

    def _copy_impl(self, new_name: Optional[str] = None) -> "SE3":
        return SE3(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SE3":
        return cast(SE3, super().copy(new_name=new_name))

    def transform_to(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        raise NotImplementedError

    def transform_from(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        raise NotImplementedError
