# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, cast

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
            dtype = data.dtype
            self._SO3_matrix_check(data)
        super().__init__(data=data, name=name, dtype=dtype)
        if quaternion is not None:
            if quaternion.ndim == 1:
                quaternion = quaternion.unsqueeze(0)
            self.update_from_unit_quaternion(quaternion)

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.eye(3, 3).view(1, 3, 3)

    def update_from_unit_quaternion(self, quaternion: torch.Tensor):
        self.update(self.unit_quaternion_to_matrix(quaternion))

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
        # _check = (
        #     torch.linalg.matmul(matrix, matrix.transpose(1, 2)) - torch.eye(3, 3)
        # ).abs().max().item() < SO3.SO3_EPS
        _check = True
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
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        raise NotImplementedError

    def _log_map_impl(self) -> torch.Tensor:
        raise NotImplementedError

    def _compose_impl(self, so3_2: LieGroup) -> "SO3":
        raise NotImplementedError

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO3":
        return SO3(data=self.data.transpose(1, 2).clone())

    def to_matrix(self) -> torch.Tensor:
        return self.data.clone()

    def to_quaternion(self) -> torch.Tensor:
        raise NotImplementedError

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
        _check = matrix.ndim == 3 and matrix.shape[1:] == (3, 3)
        _check &= (
            matrix.transpose(1, 2) + matrix
        ).abs().max().item() < theseus.constants.EPS
        if not _check:
            raise ValueError("Invalid hat matrix for SO3.")
        vec = torch.zeros(matrix.shape[0], 3, dtype=matrix.dtype, device=matrix.device)
        vec[:, 0] = matrix[:, 2, 1]
        vec[:, 1] = matrix[:, 0, 2]
        vec[:, 2] = matrix[:, 1, 0]
        return vec

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
    def unit_quaternion_to_matrix(quaternion: torch.torch.Tensor):
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
        ret = torch.zeros(quaternion.shape[0], 3, 3).to(
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
