# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import torch

from theseus.constants import DeviceType
from .vector import Vector


def _prepare_dof_and_tensor(
    expected_dof: int, tensor: Optional[torch.Tensor]
) -> Tuple[Optional[int], Optional[torch.Tensor]]:
    dof = None
    if tensor is None:
        dof = expected_dof
    else:
        if tensor.ndim == 1:
            tensor = tensor.view(1, -1)
        if tensor.shape[1] != expected_dof:
            raise ValueError(
                f"Provided tensor must have shape (batch_size, {expected_dof})."
            )
    return dof, tensor


class Point2(Vector):
    def __init__(
        self,
        tensor: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dof, tensor = _prepare_dof_and_tensor(2, tensor)
        super().__init__(dof=dof, tensor=tensor, name=name, dtype=dtype)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "Point2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return Point2(
            tensor=torch.rand(
                size[0],
                2,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "Point2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return Point2(
            tensor=torch.randn(
                size[0],
                2,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError("Point2D data tensors can only be 2D vectors.")

        return True

    def __add__(self, other: Vector) -> "Point2":
        return cast(Point2, super().__add__(other))

    def __sub__(self, other: Vector) -> "Point2":
        return cast(Point2, super().__sub__(other))

    def __mul__(self, other: Union["Vector", torch.Tensor]) -> "Point2":
        return cast(Point2, super().__mul__(other))

    def __truediv__(self, other: Union["Vector", torch.Tensor]) -> "Point2":
        return cast(Point2, super().__truediv__(other))

    def __neg__(self) -> "Point2":
        return cast(Point2, super().__neg__())

    def cat(self, vecs: Union["Vector", Tuple["Vector"], List["Vector"]]) -> "Point2":
        return cast(Point2, super().cat(vecs))

    def abs(self) -> "Point2":
        return cast(Point2, super().abs())

    def x(self) -> torch.Tensor:
        return self[:, 0]

    def y(self) -> torch.Tensor:
        return self[:, 1]

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "Point2":
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 2:
            raise ValueError("Tangent vectors of Point2 should be 2-D vectors.")

        Vector._exp_map_jacobian_impl(tangent_vector, jacobians)

        return Point2(tensor=tangent_vector.clone())

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "Point2":
        return cast(Point2, super().copy(new_name=new_name))


class Point3(Vector):
    def __init__(
        self,
        tensor: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dof, tensor = _prepare_dof_and_tensor(3, tensor)
        super().__init__(dof=dof, tensor=tensor, name=name, dtype=dtype)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "Point3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return Point3(
            tensor=torch.rand(
                size[0],
                3,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "Point3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return Point3(
            tensor=torch.randn(
                size[0],
                3,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError("Point3 data tensors can only be 3D vectors.")

        return True

    def __add__(self, other: Vector) -> "Point3":
        return cast(Point3, super().__add__(other))

    def __sub__(self, other: Vector) -> "Point3":
        return cast(Point3, super().__sub__(other))

    def __mul__(self, other: Union["Vector", torch.Tensor]) -> "Point3":
        return cast(Point3, super().__mul__(other))

    def __truediv__(self, other: Union["Vector", torch.Tensor]) -> "Point3":
        return cast(Point3, super().__truediv__(other))

    def __neg__(self) -> "Point3":
        return cast(Point3, super().__neg__())

    def cat(self, vecs: Union["Vector", Tuple["Vector"], List["Vector"]]) -> "Point3":
        return cast(Point3, super().cat(vecs))

    def abs(self) -> "Point3":
        return cast(Point3, super().abs())

    def x(self) -> torch.Tensor:
        return self[:, 0]

    def y(self) -> torch.Tensor:
        return self[:, 1]

    def z(self) -> torch.Tensor:
        return self[:, 2]

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "Point3":
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 3:
            raise ValueError("Tangent vectors of Point3 should be 3-D vectors.")

        Vector._exp_map_jacobian_impl(tangent_vector, jacobians)

        return Point3(tensor=tangent_vector.clone())

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "Point3":
        return cast(Point3, super().copy(new_name=new_name))


rand_point2 = Point2.rand
randn_point2 = Point2.randn
rand_point3 = Point3.rand
randn_point3 = Point3.randn
