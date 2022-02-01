# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

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
            if x_y_z_quaternion is not None:
                if x_y_z_quaternion.ndim == 1:
                    x_y_z_quaternion = x_y_z_quaternion.unsqueeze(0)
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
        raise NotImplementedError

    @staticmethod
    def _SE3_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 4):
            raise ValueError("SE(3) can only be 3x4 matrices.")
        SO3._SO3_matrix_check(matrix.data[:, :3, :3])

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        raise NotImplementedError

    def _log_map_impl(self) -> torch.Tensor:
        raise NotImplementedError

    def _compose_impl(self, so3_2: LieGroup) -> "SE3":
        raise NotImplementedError

    def _inverse_impl(self, get_jacobian: bool = False) -> "SE3":
        raise NotImplementedError

    def to_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def update_from_x_y_z_quaternion(self, x_y_z_quaternion: torch.Tensor):
        if x_y_z_quaternion.ndim != 2 and x_y_z_quaternion.shape[1] != 7:
            raise ValueError("x_y_z_quaternion can only be 4-D vectors.")

        batch_size = x_y_z_quaternion.shape[0]
        self.data = torch.empty(batch_size, 3, 4).to(
            device=x_y_z_quaternion.device, dtype=x_y_z_quaternion.dtype
        )
        self[:, :3, :3] = SO3.unit_quaternion_to_matrix(x_y_z_quaternion[3:])
        self[:, :, 3] = x_y_z_quaternion[:, :3]

    def update_from_rot_and_trans(
        self, rotation: SO3, translation: Union[Point3, torch.Tensor]
    ):
        raise NotImplementedError

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _transform_shape_check(self, point: Union[Point3, torch.Tensor]):
        err_msg = "SE(3) can only transform 3-D vectors."
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

    def untransform_to(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        raise NotImplementedError
