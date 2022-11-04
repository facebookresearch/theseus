# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Optional, Union, cast

import torch

import theseus.constants

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
        strict: bool = False,
    ):
        if quaternion is not None and tensor is not None:
            raise ValueError("Please provide only one of quaternion or tensor.")
        if quaternion is not None:
            dtype = quaternion.dtype
        super().__init__(tensor=tensor, name=name, dtype=dtype, strict=strict)
        if quaternion is not None:
            self.update_from_unit_quaternion(quaternion)

        self._resolve_eps()

    def _resolve_eps(self):
        self._NEAR_ZERO_EPS = theseus.constants._SO3_NEAR_ZERO_EPS[self.tensor.dtype]
        self._NEAR_PI_EPS = theseus.constants._SO3_NEAR_PI_EPS[self.tensor.dtype]

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SO3":
        # Reference:
        # https://web.archive.org/web/20211105205926/http://planning.cs.uiuc.edu/node198.html
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        u = torch.rand(
            3,
            size[0],
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        u1 = u[0]
        u2, u3 = u[1:3] * 2 * theseus.constants.PI

        a = torch.sqrt(1.0 - u1)
        b = torch.sqrt(u1)
        quaternion = torch.stack(
            [
                a * torch.sin(u2),
                a * torch.cos(u2),
                b * torch.sin(u3),
                b * torch.cos(u3),
            ],
            dim=1,
        )
        assert quaternion.shape == (size[0], 4)
        return SO3(quaternion=quaternion)

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SO3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return SO3.exp_map(
            theseus.constants.PI
            * torch.randn(
                size[0],
                3,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

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
        ret = self.tensor.new_zeros(euclidean_grad.shape[:-1])
        if is_sparse:
            temp = torch.einsum("i...jk,i...jl->i...lk", euclidean_grad, self.tensor)
        else:
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

            MATRIX_EPS = theseus.constants._SO3_MATRIX_EPS[tensor.dtype]
            if tensor.dtype != torch.float64:
                tensor = tensor.double()

            _check = (
                torch.matmul(tensor, tensor.transpose(1, 2))
                - torch.eye(3, 3, dtype=tensor.dtype, device=tensor.device)
            ).abs().max().item() < MATRIX_EPS
            _check &= (torch.linalg.det(tensor) - 1).abs().max().item() < MATRIX_EPS

        return _check

    @staticmethod
    def _unit_quaternion_check(quaternion: torch.Tensor):
        if quaternion.ndim != 2 or quaternion.shape[1] != 4:
            raise ValueError("Quaternions can only be 4-D vectors.")

        checks_enabled, silent_unchecks = _LieGroupCheckContext.get_context()
        if checks_enabled:
            QUANTERNION_EPS = theseus.constants._SO3_QUATERNION_EPS[quaternion.dtype]

            if quaternion.dtype != torch.float64:
                quaternion = quaternion.double()

            if (
                torch.linalg.norm(quaternion, dim=1) - 1
            ).abs().max().item() >= QUANTERNION_EPS:
                raise ValueError("Not unit quaternions.")
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

        checks_enabled, silent_unchecks = _LieGroupCheckContext.get_context()
        if checks_enabled:
            if (
                matrix.transpose(1, 2) + matrix
            ).abs().max().item() > theseus.constants._SO3_HAT_EPS[matrix.dtype]:
                raise ValueError("Hat matrices of SO(3) can only be skew-symmetric.")
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
            raise ValueError("Tangent vectors of SO3 should be 3-D vectors.")
        ret = SO3(dtype=tangent_vector.dtype)
        theta = torch.linalg.norm(tangent_vector, dim=1, keepdim=True).unsqueeze(1)
        theta2 = theta**2
        # Compute the approximations when theta ~ 0
        near_zero = theta < theseus.constants._SO3_NEAR_ZERO_EPS[tangent_vector.dtype]
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)

        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine = theta.sin()
        sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
        one_minus_cosie_by_theta2 = torch.where(
            near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
        )
        ret.tensor = (
            one_minus_cosie_by_theta2
            * tangent_vector.view(-1, 3, 1)
            @ tangent_vector.view(-1, 1, 3)
        )

        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        sine_axis = sine_by_theta.view(-1, 1) * tangent_vector
        ret[:, 0, 1] -= sine_axis[:, 2]
        ret[:, 1, 0] += sine_axis[:, 2]
        ret[:, 0, 2] += sine_axis[:, 1]
        ret[:, 2, 0] -= sine_axis[:, 1]
        ret[:, 1, 2] -= sine_axis[:, 0]
        ret[:, 2, 1] += sine_axis[:, 0]

        if jacobians is not None:
            SO3._check_jacobians_list(jacobians)
            theta3_nz = theta_nz * theta2_nz
            theta_minus_sine_by_theta3 = torch.where(
                near_zero, torch.zeros_like(theta), (theta - sine) / theta3_nz
            )
            jac = (
                theta_minus_sine_by_theta3
                * tangent_vector.view(-1, 3, 1)
                @ tangent_vector.view(-1, 1, 3)
            )
            diag_jac = jac.diagonal(dim1=1, dim2=2)
            diag_jac += sine_by_theta.view(-1, 1)

            jac_temp = one_minus_cosie_by_theta2.view(-1, 1) * tangent_vector

            jac[:, 0, 1] += jac_temp[:, 2]
            jac[:, 1, 0] -= jac_temp[:, 2]
            jac[:, 0, 2] -= jac_temp[:, 1]
            jac[:, 2, 0] += jac_temp[:, 1]
            jac[:, 1, 2] += jac_temp[:, 0]
            jac[:, 2, 1] -= jac_temp[:, 0]

            jacobians.append(jac)

        return ret

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
            raise ValueError("SO3 data tensors can only be 3x3 matrices.")

        U, _, V = torch.svd(tensor)
        Vtr = V.transpose(1, 2)
        S = torch.diag(
            torch.tensor([1, 1, -1], dtype=tensor.dtype, device=tensor.device)
        )
        temp = (U @ Vtr, U @ S @ Vtr)
        sign = torch.det(temp[0]).reshape([-1, 1, 1]) > 0

        return torch.where(sign, temp[0], temp[1])

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        sine_axis = self.tensor.new_zeros(self.shape[0], 3)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        cosine = 0.5 * (self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2] - 1)
        sine = sine_axis.norm(dim=1)
        theta = torch.atan2(sine, cosine)

        near_zero = theta < self._NEAR_ZERO_EPS

        near_pi = 1 + cosine <= self._NEAR_PI_EPS
        # theta != pi
        near_zero_or_near_pi = torch.logical_or(near_zero, near_pi)
        # Compute the approximation of theta / sin(theta) when theta is near to 0
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)
        sine_nz = torch.where(near_zero_or_near_pi, non_zero, sine)
        scale = torch.where(
            near_zero_or_near_pi,
            1 + sine**2 / 6,
            theta / sine_nz,
        )
        ret = sine_axis * scale.view(-1, 1)

        # # theta ~ pi
        ddiag = torch.diagonal(self.tensor, dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        aux = torch.ones(self.shape[0], dtype=torch.bool)
        sel_rows = 0.5 * (self[aux, major] + self[aux, :, major])
        sel_rows[aux, major] -= cosine
        axis = sel_rows / torch.where(
            near_zero,
            non_zero,
            sel_rows.norm(dim=1),
        ).view(-1, 1)
        sign_tmp = sine_axis[aux, major].sign()
        sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
        ret = torch.where(near_pi.view(-1, 1), axis * (theta * sign).view(-1, 1), ret)

        if jacobians is not None:
            SO3._check_jacobians_list(jacobians)

            theta2 = theta**2
            sine_theta = sine * theta
            two_cosine_minus_two = 2 * cosine - 2
            two_cosine_minus_two_nz = torch.where(
                near_zero, non_zero, two_cosine_minus_two
            )
            theta2_nz = torch.where(near_zero, non_zero, theta2)

            a = torch.where(
                near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz
            )
            b = torch.where(
                near_zero,
                1.0 / 12 + theta2 / 720,
                (sine_theta + two_cosine_minus_two)
                / (theta2_nz * two_cosine_minus_two_nz),
            )

            jac = (b.view(-1, 1) * ret).view(-1, 3, 1) * ret.view(-1, 1, 3)

            half_ret = 0.5 * ret
            jac[:, 0, 1] -= half_ret[:, 2]
            jac[:, 1, 0] += half_ret[:, 2]
            jac[:, 0, 2] += half_ret[:, 1]
            jac[:, 2, 0] -= half_ret[:, 1]
            jac[:, 1, 2] -= half_ret[:, 0]
            jac[:, 2, 1] += half_ret[:, 0]

            diag_jac = torch.diagonal(jac, dim1=1, dim2=2)
            diag_jac += a.view(-1, 1)

            jacobians.append(jac)

        return ret

    def _compose_impl(self, so3_2: LieGroup) -> "SO3":
        so3_2 = cast(SO3, so3_2)
        ret = SO3()
        ret.tensor = self.tensor @ so3_2.tensor
        return ret

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO3":
        # if self.tensor is a valid SO(3), then self.tensor.transpose(1, 2)
        # must be valid as well
        return SO3(tensor=self.tensor.transpose(1, 2).clone(), strict=False)

    def to_matrix(self) -> torch.Tensor:
        return self.tensor.clone()

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self) -> torch.Tensor:
        sine_axis = self.tensor.new_zeros(self.shape[0], 3)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        w = 0.5 * (1 + self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2]).clamp(0, 4).sqrt()

        near_zero = w > 1 - self._NEAR_ZERO_EPS
        near_pi = w <= self._NEAR_PI_EPS
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
        sine_half_theta = (0.5 * (1 - cosine_near_pi)).clamp(0, 1).sqrt().view(-1, 1)
        ret[:, 1:] = torch.where(
            near_pi.view(-1, 1), axis * sine_half_theta, ret[:, 1:]
        )

        return ret

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (3, 1)
        _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 3
        if not _check:
            raise ValueError("Invalid vee matrix for SO3.")
        matrix = tangent_vector.new_zeros(tangent_vector.shape[0], 3, 3)

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
        err_msg = (
            f"SO3 can only rotate vectors of shape [{self.shape[0]}, 3] or [1, 3], "
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

    # The quaternion takes the [w x y z] convention
    @staticmethod
    def unit_quaternion_to_SO3(quaternion: torch.Tensor) -> "SO3":
        if quaternion.ndim == 1:
            quaternion = quaternion.unsqueeze(0)
        SO3._unit_quaternion_check(quaternion)

        w = quaternion[:, 0]
        x = quaternion[:, 1]
        y = quaternion[:, 2]
        z = quaternion[:, 3]
        q00 = w * w
        q01 = w * x
        q02 = w * y
        q03 = w * z
        q11 = x * x
        q12 = x * y
        q13 = x * z
        q22 = y * y
        q23 = y * z
        q33 = z * z

        ret = SO3()
        ret.tensor = quaternion.new_zeros(quaternion.shape[0], 3, 3)

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
        # if self.tensor is a valid SO(3), so is the copy
        return SO3(tensor=self.tensor.clone(), name=new_name, strict=False)

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
            p = point.tensor.view(-1, 3, 1)

        ret = Point3(tensor=(self.tensor @ p).view(-1, 3))
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            # Right jacobians for SO(3) are computed
            Jrot = -self.tensor @ SO3.hat(p)
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
            p = point.tensor.view(-1, 3, 1)

        ret = Point3(tensor=(self.tensor.transpose(1, 2) @ p).view(-1, 3))
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            # Left jacobians for SO3 are computed
            Jrot = self.tensor.new_zeros(batch_size, 3, 3)
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

    def _deprecated_log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        sine_axis = self.tensor.new_zeros(self.shape[0], 3)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        cosine = 0.5 * (self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2] - 1)
        sine = sine_axis.norm(dim=1)
        theta = torch.atan2(sine, cosine)

        near_zero = theta < self._NEAR_ZERO_EPS

        near_pi = 1 + cosine <= self._NEAR_PI_EPS
        # theta != pi
        near_zero_or_near_pi = torch.logical_or(near_zero, near_pi)
        # Compute the approximation of theta / sin(theta) when theta is near to 0
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)
        sine_nz = torch.where(near_zero_or_near_pi, non_zero, sine)
        scale = torch.where(
            near_zero_or_near_pi,
            1 + sine**2 / 6,
            theta / sine_nz,
        )
        ret = sine_axis * scale.view(-1, 1)

        if near_pi.any():
            ddiag = torch.diagonal(self[near_pi], dim1=1, dim2=2)
            # Find the index of major coloumns and diagonals
            major = torch.logical_and(
                ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
            ) + 2 * torch.logical_and(
                ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1]
            )
            sel_rows = 0.5 * (self[near_pi, major] + self[near_pi, :, major])
            aux = torch.ones(sel_rows.shape[0], dtype=torch.bool)
            sel_rows[aux, major] -= cosine[near_pi]
            axis = sel_rows / sel_rows.norm(dim=1, keepdim=True)
            sign_tmp = sine_axis[near_pi, major].sign()
            sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
            ret[near_pi] = axis * (theta[near_pi] * sign).view(-1, 1)

        if jacobians is not None:
            SO3._check_jacobians_list(jacobians)

            theta2 = theta**2
            sine_theta = sine * theta
            two_cosine_minus_two = 2 * cosine - 2
            two_cosine_minus_two_nz = torch.where(
                near_zero, non_zero, two_cosine_minus_two
            )
            theta2_nz = torch.where(near_zero, non_zero, theta2)

            a = torch.where(
                near_zero, 1 - theta2 / 12, -sine_theta / two_cosine_minus_two_nz
            )
            b = torch.where(
                near_zero,
                1.0 / 12 + theta2 / 720,
                (sine_theta + two_cosine_minus_two)
                / (theta2_nz * two_cosine_minus_two_nz),
            )

            jac = (b.view(-1, 1) * ret).view(-1, 3, 1) * ret.view(-1, 1, 3)

            half_ret = 0.5 * ret
            jac[:, 0, 1] -= half_ret[:, 2]
            jac[:, 1, 0] += half_ret[:, 2]
            jac[:, 0, 2] += half_ret[:, 1]
            jac[:, 2, 0] -= half_ret[:, 1]
            jac[:, 1, 2] -= half_ret[:, 0]
            jac[:, 2, 1] += half_ret[:, 0]

            diag_jac = torch.diagonal(jac, dim1=1, dim2=2)
            diag_jac += a.view(-1, 1)

            jacobians.append(jac)

        return ret

    def _deprecated_to_quaternion(self) -> torch.Tensor:
        sine_axis = self.tensor.new_zeros(self.shape[0], 3)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        w = 0.5 * (1 + self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2]).clamp(0, 4).sqrt()

        near_pi = w <= self._NEAR_PI_EPS
        non_zero = self.tensor.new_ones([1])

        ret = self.tensor.new_zeros(self.shape[0], 4)
        # theta != pi
        ret[:, 0] = w
        ret[:, 1:] = 0.5 * sine_axis / torch.where(near_pi, non_zero, w).view(-1, 1)

        # theta ~ pi
        ddiag = torch.diagonal(self[near_pi], dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        sel_rows = 0.5 * (self[near_pi, major] + self[near_pi, :, major])
        aux = torch.ones(sel_rows.shape[0], dtype=torch.bool)
        cosine_near_pi = 0.5 * (
            self[near_pi, 0, 0] + self[near_pi, 1, 1] + self[near_pi, 2, 2] - 1
        )
        sel_rows[aux, major] -= cosine_near_pi
        axis = (
            sel_rows
            / sel_rows.norm(dim=1, keepdim=True)
            * torch.where(
                sine_axis[near_pi, major].view(-1, 1) >= 0, non_zero, -non_zero
            )
        )
        sine_half_theta = (0.5 * (1 - cosine_near_pi)).clamp(0, 1).sqrt().view(-1, 1)
        ret[near_pi, 1:] = axis * sine_half_theta

        return ret


rand_so3 = SO3.rand
randn_so3 = SO3.randn
