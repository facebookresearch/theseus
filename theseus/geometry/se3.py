# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union, cast

import torch

import theseus.constants
from torchlie.functional import SE3 as SE3_base

from .lie_group import LieGroup
from .lie_group_check import _LieGroupCheckContext, no_lie_group_check
from .point_types import Point3
from .so3 import SO3


class SE3(LieGroup):
    def __init__(
        self,
        x_y_z_quaternion: Optional[torch.Tensor] = None,
        tensor: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        strict_checks: bool = False,
        disable_checks: bool = False,
    ):
        if x_y_z_quaternion is not None and tensor is not None:
            raise ValueError("Please provide only one of x_y_z_quaternion or tensor.")
        if x_y_z_quaternion is not None:
            dtype = x_y_z_quaternion.dtype
        super().__init__(
            tensor=tensor,
            name=name,
            dtype=dtype,
            strict_checks=strict_checks,
            disable_checks=disable_checks,
        )
        if x_y_z_quaternion is not None:
            self.update_from_x_y_z_quaternion(x_y_z_quaternion=x_y_z_quaternion)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: theseus.constants.DeviceType = None,
        requires_grad: bool = False,
    ) -> "SE3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        tensor = SE3_base.rand(
            *size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return SE3(tensor=tensor, disable_checks=True)

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: theseus.constants.DeviceType = None,
        requires_grad: bool = False,
    ) -> "SE3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        tensor = SE3_base.randn(
            *size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return SE3(tensor=tensor, disable_checks=True)

    @staticmethod
    def _init_tensor() -> torch.Tensor:  # type: ignore
        return torch.eye(3, 4).view(1, 3, 4)

    def dof(self) -> int:
        return 6

    def __repr__(self) -> str:
        return f"SE3(tensor={self.tensor}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            return f"SE3(matrix={self.tensor}), name={self.name})"

    def _adjoint_impl(self) -> torch.Tensor:
        return SE3_base.adj(self.tensor)

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)
        if is_sparse:
            return SE3_base.left_project(self.tensor, euclidean_grad)
        else:
            ret = self.tensor.new_zeros(euclidean_grad.shape[:-2] + torch.Size([6]))
            temp = torch.einsum(
                "...jk,...ji->...ik", euclidean_grad, self.tensor[:, :, :3]
            )
            ret[..., :3] = temp[..., 3]
            ret[..., 3] = temp[..., 2, 1] - temp[..., 1, 2]
            ret[..., 4] = temp[..., 0, 2] - temp[..., 2, 0]
            ret[..., 5] = temp[..., 1, 0] - temp[..., 0, 1]

            return ret

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 4):
            raise ValueError("SE3 data tensors can only be 3x4 matrices.")
        try:
            SE3_base.check_group_tensor(tensor)
        except ValueError:
            return False
        return True

    @staticmethod
    def x_y_z_unit_quaternion_to_SE3(x_y_z_quaternion: torch.Tensor) -> "SE3":
        if x_y_z_quaternion.ndim == 1:
            x_y_z_quaternion = x_y_z_quaternion.unsqueeze(0)

        if x_y_z_quaternion.ndim != 2 and x_y_z_quaternion.shape[1] != 7:
            raise ValueError("x_y_z_quaternion can only be 7-D vectors.")

        ret = SE3()

        batch_size = x_y_z_quaternion.shape[0]
        ret.tensor = torch.empty(batch_size, 3, 4).to(
            device=x_y_z_quaternion.device, dtype=x_y_z_quaternion.dtype
        )
        ret[:, :, :3] = SO3.unit_quaternion_to_SO3(x_y_z_quaternion[:, 3:]).tensor
        ret[:, :, 3] = x_y_z_quaternion[:, :3]

        return ret

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
            raise ValueError("Hat matrices of SE3 can only be 4x4 matrices")

        checks_enabled, silent_unchecks, _ = _LieGroupCheckContext.get_context()
        if checks_enabled:
            return SE3_base.check_hat_tensor(matrix)
        elif not silent_unchecks:
            warnings.warn(
                "Lie group checks are disabled, so the skew-symmetry of hat matrices is "
                "not checked for SE3.",
                RuntimeWarning,
            )

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "SE3":
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 6:
            raise ValueError("Tangent vectors of SE3 can only be 6D vectors.")
        return SE3(
            tensor=SE3_base.exp(tangent_vector, jacobians=jacobians),
            disable_checks=True,
        )

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 4):
            raise ValueError("SE3 data tensors can only be 3x4 matrices.")
        return SE3_base.normalize(tensor)

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return SE3_base.log(self.tensor, jacobians=jacobians)

    def _compose_impl(self, se3_2: LieGroup) -> "SE3":
        return SE3(
            tensor=SE3_base.compose(self.tensor, se3_2.tensor), strict_checks=False
        )

    def _inverse_impl(self, get_jacobian: bool = False) -> "SE3":
        return SE3(tensor=SE3_base.inv(self.tensor), disable_checks=True)

    def to_matrix(self) -> torch.Tensor:
        ret = self.tensor.new_zeros(self.shape[0], 4, 4)
        ret[:, :3] = self.tensor
        ret[:, 3, 3] = 1
        return ret

    # The quaternion takes the [w x y z] convention
    def update_from_x_y_z_quaternion(self, x_y_z_quaternion: torch.Tensor):
        self.update(SE3.x_y_z_unit_quaternion_to_SE3(x_y_z_quaternion))

    def update_from_rot_and_trans(self, rotation: SO3, translation: Point3):
        if rotation.shape[0] != translation.shape[0]:
            raise ValueError("rotation and translation must have the same batch size.")

        if rotation.dtype != translation.dtype:
            raise ValueError("rotation and translation must be of the same type.")

        if rotation.device != translation.device:
            raise ValueError("rotation and translation must be on the same device.")

        self.tensor = torch.cat(
            (rotation.tensor, translation.tensor.unsqueeze(2)), dim=2
        )

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        return SE3_base.hat(tangent_vector)

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        return SE3_base.vee(matrix)

    def _copy_impl(self, new_name: Optional[str] = None) -> "SE3":
        # if self.tensor is a valid SE3, so is the copy
        return SE3(tensor=self.tensor.clone(), name=new_name, disable_checks=True)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SE3":
        return cast(SE3, super().copy(new_name=new_name))

    def _transform_shape_check(self, point: Union[Point3, torch.Tensor]):
        err_msg = (
            f"SE3 can only transform vectors of shape [{self.shape[0]}, 3] or [1, 3], "
            f"but the input has shape {point.shape}."
        )

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
            raise ValueError(err_msg)

    def transform_from(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._transform_shape_check(point)
        p = point if isinstance(point, torch.Tensor) else point.tensor
        return Point3(SE3_base.transform(self.tensor, p, jacobians=jacobians))

    def transform_to(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._transform_shape_check(point)
        p = point if isinstance(point, torch.Tensor) else point.tensor
        return Point3(SE3_base.untransform(self.tensor, p, jacobians=jacobians))

    # The returned tensor will have 7 elements, [x, y, z, qw, qx, qy, qz] where
    # [x y z] corresponds to the translation and [qw qx qy qz] to the quaternion
    # using the [w x y z] convention
    def to_x_y_z_quaternion(self) -> torch.Tensor:
        ret = self.tensor.new_zeros(self.shape[0], 7)
        ret[:, :3] = self.tensor[:, :, 3]
        ret[:, 3:] = SO3(
            tensor=self.tensor[:, :, :3], disable_checks=True
        ).to_quaternion()
        return ret

    def rotation(self) -> SO3:
        return SO3(tensor=self.tensor[:, :, :3], disable_checks=True)

    def translation(self) -> Point3:
        with no_lie_group_check(silent=True):
            return Point3(tensor=self.tensor[:, :, 3].view(-1, 3))

    # calls to() on the internal tensors
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)


rand_se3 = SE3.rand
randn_se3 = SE3.randn
