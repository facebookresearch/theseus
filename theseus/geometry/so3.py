# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union, cast

import torch

import theseus.constants
from theseus.global_params import _THESEUS_GLOBAL_PARAMS
from torchlie.functional import SO3 as SO3_base

from .lie_group import LieGroup
from .lie_group_check import _LieGroupCheckContext
from .point_types import Point3


class SO3(LieGroup):
    def __init__(
        self,
        quaternion: Optional[torch.Tensor] = None,
        tensor: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        strict_checks: bool = False,
        disable_checks: bool = False,
    ):
        if quaternion is not None and tensor is not None:
            raise ValueError("Please provide only one of quaternion or tensor.")
        if quaternion is not None:
            dtype = quaternion.dtype
        super().__init__(
            tensor=tensor,
            name=name,
            dtype=dtype,
            strict_checks=strict_checks,
            disable_checks=disable_checks,
        )
        if quaternion is not None:
            self.update_from_unit_quaternion(quaternion)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: theseus.constants.DeviceType = None,
        requires_grad: bool = False,
    ) -> "SO3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        tensor = SO3_base.rand(
            *size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return SO3(tensor=tensor, disable_checks=True)

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: theseus.constants.DeviceType = None,
        requires_grad: bool = False,
    ) -> "SO3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        tensor = SO3_base.randn(
            *size,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return SO3(tensor=tensor, disable_checks=True)

    @staticmethod
    def _init_tensor() -> torch.Tensor:  # type: ignore
        return torch.eye(3, 3).view(1, 3, 3)

    def update_from_unit_quaternion(self, quaternion: torch.Tensor):
        self.update(self.unit_quaternion_to_SO3(quaternion))

    def dof(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"SO3(tensor={self.tensor}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            return f"SO3(matrix={self.tensor}), name={self.name})"

    def _adjoint_impl(self) -> torch.Tensor:
        return self.tensor.clone()

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)
        if is_sparse:
            return SO3_base.left_project(self.tensor, euclidean_grad)
        else:
            ret = self.tensor.new_zeros(euclidean_grad.shape[:-1])
            temp = torch.einsum("...jk,...ji->...ik", euclidean_grad, self.tensor)
            ret[..., 0] = temp[..., 2, 1] - temp[..., 1, 2]
            ret[..., 1] = temp[..., 0, 2] - temp[..., 2, 0]
            ret[..., 2] = temp[..., 1, 0] - temp[..., 0, 1]
            return ret

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        with torch.no_grad():
            if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
                raise ValueError("SO3 data tensors can only be 3x3 matrices.")
            try:
                SO3_base.check_group_tensor(tensor)
            except ValueError:
                return False
        return True

    @staticmethod
    def _unit_quaternion_check(quaternion: torch.Tensor):
        if quaternion.ndim != 2 or quaternion.shape[1] != 4:
            raise ValueError("Quaternions can only be 4-D vectors.")

        checks_enabled, silent_unchecks, _ = _LieGroupCheckContext.get_context()
        if checks_enabled:
            SO3_base.check_unit_quaternion(quaternion)
        elif not silent_unchecks:
            warnings.warn(
                "Lie group checks are disabled, so the validness of unit quaternions is not "
                "checked for SO3.",
                RuntimeWarning,
            )

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
            raise ValueError("Hat matrices of SO(3) can only be 3x3 matrices")

        checks_enabled, silent_unchecks, _ = _LieGroupCheckContext.get_context()
        if checks_enabled:
            SO3_base.check_hat_tensor(matrix)
        elif not silent_unchecks:
            warnings.warn(
                "Lie group checks are disabled, so the skew-symmetry of hat matrices is "
                "not checked for SO3.",
                RuntimeWarning,
            )

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "SO3":
        if tangent_vector.ndim != 2 or tangent_vector.shape[1] != 3:
            raise ValueError("Tangent vectors of SO3 should be batched 3-D vectors.")
        return SO3(
            tensor=SO3_base.exp(tangent_vector, jacobians=jacobians),
            disable_checks=True,
        )

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
            raise ValueError("SO3 data tensors can only be batched 3x3 matrices.")
        return SO3_base.normalize(tensor)

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return SO3_base.log(self.tensor, jacobians=jacobians)

    def _compose_impl(self, so3_2: LieGroup) -> "SO3":
        return SO3(
            tensor=SO3_base.compose(self.tensor, so3_2.tensor), strict_checks=False
        )

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO3":
        # if self.tensor is a valid SO(3), then self.tensor.transpose(1, 2)
        # must be valid as well
        return SO3(tensor=self.tensor.transpose(1, 2).clone(), disable_checks=True)

    def to_matrix(self) -> torch.Tensor:
        return self.tensor.clone()

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self) -> torch.Tensor:
        sine_axis = self.tensor.new_zeros(self.shape[0], 3)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        w = 0.5 * (1 + self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2]).clamp(0, 4).sqrt()

        near_zero = w > 1 - _THESEUS_GLOBAL_PARAMS.get_eps("so3", "near_zero", w.dtype)
        near_pi = w <= _THESEUS_GLOBAL_PARAMS.get_eps("so3", "near_pi", w.dtype)
        non_zero = self.tensor.new_ones([1])

        ret = self.tensor.new_zeros(self.shape[0], 4)
        # theta != pi
        ret[:, 0] = w
        ret[:, 1:] = 0.5 * sine_axis / torch.where(near_pi, non_zero, w).view(-1, 1)

        # theta ~ pi
        ddiag = torch.diagonal(self.tensor, dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        aux = torch.ones(self.shape[0], dtype=torch.bool)
        sel_rows = 0.5 * (self[aux, major] + self[aux, :, major])
        cosine_near_pi = 0.5 * (self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2] - 1)
        sel_rows[aux, major] -= cosine_near_pi
        axis = (
            sel_rows
            / torch.where(
                near_zero.view(-1, 1),
                non_zero.view(-1, 1),
                sel_rows.norm(dim=1, keepdim=True),
            )
            * torch.where(sine_axis[aux, major].view(-1, 1) >= 0, non_zero, -non_zero)
        )
        sqrt_eps = _THESEUS_GLOBAL_PARAMS.get_eps("so3", "to_quaternion_sqrt", w.dtype)
        sine_half_theta = (
            (0.5 * (1 - cosine_near_pi)).clamp(sqrt_eps, 1).sqrt().view(-1, 1)
        )
        ret[:, 1:] = torch.where(
            near_pi.view(-1, 1), axis * sine_half_theta, ret[:, 1:]
        )

        return ret

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        return SO3_base.hat(tangent_vector)

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        SO3._hat_matrix_check(matrix)
        return SO3_base.vee(matrix)

    def _rotate_shape_check(self, point: Union[Point3, torch.Tensor]):
        err_msg = (
            f"Point tensor to rotate must have final dimensions of shape (3,) "
            f"or (3, 1), and be broadcastable with SO3, but received a tensor with "
            f"shape {point.shape}, while SO3 shape is {self.shape}."
        )
        if isinstance(point, torch.Tensor):
            if (
                point.ndim not in [2, 3]
                or point.shape[1] != 3
                or (point.ndim == 3 and point.shape[-1] != 1)
            ):
                raise ValueError(err_msg)

        elif point.dof() != 3:
            raise ValueError(err_msg)
        if (
            point.shape[0] != self.shape[0]
            and point.shape[0] != 1
            and self.shape[0] != 1
        ):
            raise ValueError(err_msg)

    # The quaternion takes the [w x y z] convention
    @staticmethod
    def unit_quaternion_to_SO3(quaternion: torch.Tensor) -> "SO3":
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        return SO3(
            tensor=SO3_base.quaternion_to_rotation(quaternion), disable_checks=True
        )

    def _copy_impl(self, new_name: Optional[str] = None) -> "SO3":
        # if self.tensor is a valid SO(3), so is the copy
        return SO3(tensor=self.tensor.clone(), name=new_name, disable_checks=True)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SO3":
        return cast(SO3, super().copy(new_name=new_name))

    def rotate(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._rotate_shape_check(point)
        p = point if isinstance(point, torch.Tensor) else point.tensor
        return Point3(tensor=SO3_base.transform(self.tensor, p, jacobians=jacobians))

    def unrotate(
        self,
        point: Union[Point3, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point3:
        self._rotate_shape_check(point)
        p = point if isinstance(point, torch.Tensor) else point.tensor
        return Point3(tensor=SO3_base.untransform(self.tensor, p, jacobians=jacobians))


rand_so3 = SO3.rand
randn_so3 = SO3.randn
