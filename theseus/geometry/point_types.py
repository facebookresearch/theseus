# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, cast
from xmlrpc.client import Boolean

import torch

from .lie_group import LieGroup
from .vector import Vector


def _prepare_dof_and_data(
    expected_dof: int, data: Optional[torch.Tensor]
) -> Tuple[Optional[int], Optional[torch.Tensor]]:
    dof = None
    if data is None:
        dof = expected_dof
    else:
        if data.ndim == 1:
            data = data.view(1, -1)
        if data.shape[1] != expected_dof:
            raise ValueError(
                f"Provied data tensor must have shape (batch_size, {expected_dof})."
            )
    return dof, data


class Point2(Vector):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dof, data = _prepare_dof_and_data(2, data)
        super().__init__(dof=dof, data=data, name=name, dtype=dtype)

    @staticmethod
    def rand(
        *size,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Boolean = False,
    ) -> "Point2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return Point2(
            data=torch.rand(
                size[0], 2, dtype=dtype, device=device, requires_grad=requires_grad
            )
        )

    def x(self) -> torch.Tensor:
        return self[:, 0]

    def y(self) -> torch.Tensor:
        return self[:, 1]

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        return Point2(data=tangent_vector.clone())

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "Point2":
        return cast(Point2, super().copy(new_name=new_name))


class Point3(Vector):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dof, data = _prepare_dof_and_data(3, data)
        super().__init__(dof=dof, data=data, name=name, dtype=dtype)

    @staticmethod
    def rand(
        *size,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Boolean = False,
    ) -> "Point3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return Point3(
            data=torch.rand(
                size[0], 3, dtype=dtype, device=device, requires_grad=requires_grad
            )
        )

    def x(self) -> torch.Tensor:
        return self[:, 0]

    def y(self) -> torch.Tensor:
        return self[:, 1]

    def z(self) -> torch.Tensor:
        return self[:, 2]

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        return Point3(data=tangent_vector.clone())

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "Point3":
        return cast(Point3, super().copy(new_name=new_name))
