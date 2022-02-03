# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

import theseus.constants

from .lie_group import LieGroup
from .point_types import Point3


class SO3(LieGroup):
    SO3_EPS = 5e-7

    def __init__(
        self,
        quaternion: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if quaternion is not None and data is not None:
            raise ValueError("Please provide only one of quaternion or data.")
        if quaternion is not None:
            dtype = quaternion.dtype
        if data is not None:
            self._SO3_matrix_check(data)
        super().__init__(data=data, name=name, dtype=dtype)
        if quaternion is not None:
            self.update_from_unit_quaternion(quaternion)

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.eye(3, 3).view(1, 3, 3)

    def update_from_unit_quaternion(self, quaternion: torch.Tensor):
        self.update(self.unit_quaternion_to_SO3(quaternion))

    def dof(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"SO3(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            return f"SO3(matrix={self.data}), name={self.name})"

    def _adjoint_impl(self) -> torch.Tensor:
        return self.data.clone()

    @staticmethod
    def _SO3_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
            raise ValueError("3D rotations can only be 3x3 matrices.")
        _check = (
            torch.matmul(matrix, matrix.transpose(1, 2))
            - torch.eye(3, 3, dtype=matrix.dtype, device=matrix.device)
        ).abs().max().item() < SO3.SO3_EPS
        _check &= (torch.linalg.det(matrix) - 1).abs().max().item() < SO3.SO3_EPS

        if not _check:
            raise ValueError("Not valid 3D rotations.")

    @staticmethod
    def _unit_quaternion_check(quaternion: torch.Tensor):
        if quaternion.ndim != 2 or quaternion.shape[1] != 4:
            raise ValueError("Quaternions can only be 4-D vectors.")

        if (torch.linalg.norm(quaternion, dim=1) - 1).abs().max().item() >= SO3.SO3_EPS:
            raise ValueError("Not unit quaternions.")

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
            raise ValueError("Hat matrices of SO(3) can only be 3x3 matrices")

        if (matrix.transpose(1, 2) + matrix).abs().max().item() > theseus.constants.EPS:
            raise ValueError("Hat matrices of SO(3) can only be skew-symmetric.")

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 3:
            raise ValueError("Invalid input for SO3.exp_map.")
        ret = SO3(dtype=tangent_vector.dtype)
        theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
        theta2 = theta**2
        # Compute the approximations when theta ~ 0
        small_theta = theta < 0.005
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(small_theta, non_zero, theta)
        theta2_nz = torch.where(small_theta, non_zero, theta2)
        cosine = torch.where(small_theta, 8 / (4 + theta2) - 1, theta.cos())
        sine_by_theta = torch.where(
            small_theta, 0.5 * cosine + 0.5, theta.sin() / theta_nz
        )
        one_minus_cosie_by_theta2 = torch.where(
            small_theta, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
        )
        ret.data = (
            one_minus_cosie_by_theta2
            * tangent_vector.view(-1, 3, 1)
            @ tangent_vector.view(-1, 1, 3)
        )
        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        temp = sine_by_theta.view(-1, 1) * tangent_vector
        ret[:, 0, 1] -= temp[:, 2]
        ret[:, 1, 0] += temp[:, 2]
        ret[:, 0, 2] += temp[:, 1]
        ret[:, 2, 0] -= temp[:, 1]
        ret[:, 1, 2] -= temp[:, 0]
        ret[:, 2, 1] += temp[:, 0]
        return ret

    def _log_map_impl(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 3, dtype=self.dtype, device=self.device)
        ret[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        ret[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        ret[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        cosine = 0.5 * (self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2] - 1)
        sine = ret.norm(dim=1)
        theta = torch.atan2(sine, cosine)
        # theta != pi
        not_near_pi = 1 + cosine > 1e-7
        # Compute the approximation of theta / sin(theta) when theta is near to 0
        small_theta = theta[not_near_pi] < 5e-3
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)
        sine_nz = torch.where(small_theta, non_zero, sine[not_near_pi])
        scale = torch.where(
            small_theta, 1 + sine[not_near_pi] ** 2 / 6, theta[not_near_pi] / sine_nz
        )
        ret[not_near_pi] *= scale.view(-1, 1)
        # theta ~ pi
        near_pi = ~not_near_pi
        ddiag = torch.diagonal(self[near_pi], dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        ret[near_pi] = self[near_pi, major]
        ret[near_pi, major] -= cosine[near_pi]
        ret[near_pi] *= (theta[near_pi] ** 2 / (1 - cosine[near_pi])).view(-1, 1)
        ret[near_pi] /= ret[near_pi, major].sqrt().view(-1, 1)
        return ret

    def _compose_impl(self, so3_2: LieGroup) -> "SO3":
        so3_2 = cast(SO3, so3_2)
        ret = SO3()
        ret.data = self.data @ so3_2.data
        return ret

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO3":
        return SO3(data=self.data.transpose(1, 2).clone())

    def to_matrix(self) -> torch.Tensor:
        return self.data.clone()

    def to_quaternion(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 4, dtype=self.dtype, device=self.device)
        # ret[:, 4] computes the cos(0.5 * theta)
        ret[:, 0] = (
            0.5 * (1 + self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2]).clamp(0, 4).sqrt()
        )
        # theta != pi
        not_near_pi = ret[:, 0] > 1e-5
        ret[not_near_pi, 1] = (
            0.25
            * (self[not_near_pi, 2, 1] - self[not_near_pi, 1, 2])
            / ret[not_near_pi, 0]
        )
        ret[not_near_pi, 2] = (
            0.25
            * (self[not_near_pi, 0, 2] - self[not_near_pi, 2, 0])
            / ret[not_near_pi, 0]
        )
        ret[not_near_pi, 3] = (
            0.25
            * (self[not_near_pi, 1, 0] - self[not_near_pi, 0, 1])
            / ret[not_near_pi, 0]
        )
        # theta ~ pi
        near_pi = ~not_near_pi
        ddiag = torch.diagonal(self[near_pi], dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        ret[near_pi, 1:] = self[near_pi, major]
        cosine_near_pi = 0.5 * (
            self[near_pi, 0, 0] + self[near_pi, 1, 1] + self[near_pi, 2, 2] - 1
        )
        ret[near_pi, major + 1] -= cosine_near_pi
        ret[near_pi, 1:] /= ret[near_pi, 1:].norm(dim=1, keepdim=True)
        ret[near_pi, 1:] *= (0.5 * (1 - cosine_near_pi)).clamp(0, 1).sqrt().view(-1, 1)
        ret /= ret.norm(dim=1, keepdim=True)
        return ret

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (3, 1)
        _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 3
        if not _check:
            raise ValueError("Invalid vee matrix for SO3.")
        matrix = torch.zeros(tangent_vector.shape[0], 3, 3).to(
            dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        matrix[:, 0, 1] = -tangent_vector[:, 2].view(-1)
        matrix[:, 0, 2] = tangent_vector[:, 1].view(-1)
        matrix[:, 1, 2] = -tangent_vector[:, 0].view(-1)
        matrix[:, 1, 0] = tangent_vector[:, 2].view(-1)
        matrix[:, 2, 0] = -tangent_vector[:, 1].view(-1)
        matrix[:, 2, 1] = tangent_vector[:, 0].view(-1)
        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        SO3._hat_matrix_check(matrix)
        return torch.stack((matrix[:, 2, 1], matrix[:, 0, 2], matrix[:, 1, 0]), dim=1)

    def _rotate_shape_check(self, point: Union[Point3, torch.Tensor]):
        err_msg = "SO3 can only rotate 3-D vectors."
        if isinstance(point, torch.Tensor):
            if not point.ndim == 2 or point.shape[1] != 3:
                raise ValueError(err_msg)
        elif point.dof() != 3:
            raise ValueError(err_msg)
        if (
            point.shape[0] != self.shape[0]
            and point.shape[0] != 1
            and self.shape[0] != 1
        ):
            raise ValueError(
                "Input point batch size is not broadcastable with group batch size."
            )

    @staticmethod
    def unit_quaternion_to_SO3(quaternion: torch.torch.Tensor) -> "SO3":
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        SO3._unit_quaternion_check(quaternion)

        q0 = quaternion[:, 0]
        q1 = quaternion[:, 1]
        q2 = quaternion[:, 2]
        q3 = quaternion[:, 3]
        q00 = q0 * q0
        q01 = q0 * q1
        q02 = q0 * q2
        q03 = q0 * q3
        q11 = q1 * q1
        q12 = q1 * q2
        q13 = q1 * q3
        q22 = q2 * q2
        q23 = q2 * q3
        q33 = q3 * q3

        ret = SO3()
        ret.data = torch.zeros(quaternion.shape[0], 3, 3).to(
            dtype=quaternion.dtype, device=quaternion.device
        )
        ret[:, 0, 0] = 2 * (q00 + q11) - 1
        ret[:, 0, 1] = 2 * (q12 - q03)
        ret[:, 0, 2] = 2 * (q13 + q02)
        ret[:, 1, 0] = 2 * (q12 + q03)
        ret[:, 1, 1] = 2 * (q00 + q22) - 1
        ret[:, 1, 2] = 2 * (q23 - q01)
        ret[:, 2, 0] = 2 * (q13 - q02)
        ret[:, 2, 1] = 2 * (q23 + q01)
        ret[:, 2, 2] = 2 * (q00 + q33) - 1
        return ret

    def _copy_impl(self, new_name: Optional[str] = None) -> "SO3":
        return SO3(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SO3":
        return cast(SO3, super().copy(new_name=new_name))

    def rotate(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._rotate_shape_check(point)
        batch_size = max(self.shape[0], point.shape[0])
        if isinstance(point, torch.Tensor):
            p = point.view(-1, 3, 1)
        else:
            p = point.data.view(-1, 3, 1)

        ret = Point3(data=(self.data @ p).view(-1, 3))
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            # Right jacobians for SO(3) are computed
            Jrot = -self.data @ SO3.hat(p)
            # Jacobians for point
            Jpnt = self.to_matrix().expand(batch_size, 3, 3)

            jacobians.extend([Jrot, Jpnt])

        return ret

    def unrotate(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._rotate_shape_check(point)
        batch_size = max(self.shape[0], point.shape[0])
        if isinstance(point, torch.Tensor):
            p = point.view(-1, 3, 1)
        else:
            p = point.data.view(-1, 3, 1)

        ret = Point3(data=(self.data.transpose(1, 2) @ p).view(-1, 3))
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            # Left jacobians for SO3 are computed
            Jrot = torch.zeros(batch_size, 3, 3, dtype=self.dtype, device=self.device)
            Jrot[:, 0, 1] = -ret[:, 2]
            Jrot[:, 1, 0] = ret[:, 2]
            Jrot[:, 0, 2] = ret[:, 1]
            Jrot[:, 2, 0] = -ret[:, 1]
            Jrot[:, 1, 2] = -ret[:, 0]
            Jrot[:, 2, 1] = ret[:, 0]
            # Jacobians for point
            Jpnt = self.to_matrix().transpose(1, 2).expand(batch_size, 3, 3)

            jacobians.extend([Jrot, Jpnt])

        return ret
