# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

from .lie_group import LieGroup
from .point_types import Point3


class SE3(LieGroup):
    def __init__(
        self,
        x_y_z_quaternion: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        raise NotImplementedError

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
        raise NotImplementedError

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

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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

    def untransform_to(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        raise NotImplementedError
