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
    def __init__(
        self,
        quaternion: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        strict: bool = True,
    ):
        if quaternion is not None and data is not None:
            raise ValueError("Please provide only one of quaternion or data.")
        if quaternion is not None:
            dtype = quaternion.dtype
        if data is not None:
            if strict:
                self._data_check(data)
            elif not SO3._data_check_impl(data):
                data = SO3.normalize(data)
                raise Warning(
                    "The input data is not valid for SO3 and has been normalized."
                )
        super().__init__(data=data, name=name, dtype=dtype)
        if quaternion is not None:
            self.update_from_unit_quaternion(quaternion)

        self._resolve_eps()

    def _resolve_eps(self):
        self._NEAR_ZERO_EPS = theseus.constants._SO3_NEAR_ZERO_EPS[self.data.dtype]
        self._NEAR_PI_EPS = theseus.constants._SO3_NEAR_PI_EPS[self.data.dtype]

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SO3":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return SO3.exp_map(
            2
            * theseus.constants.PI
            * torch.rand(
                size[0],
                3,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
            - theseus.constants.PI
        )

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

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)
        ret = torch.zeros(
            euclidean_grad.shape[:-1], dtype=self.dtype, device=self.device
        )
        if is_sparse:
            temp = torch.einsum("i...jk,i...jl->i...lk", euclidean_grad, self.data)
        else:
            temp = torch.einsum("...jk,...ji->...ik", euclidean_grad, self.data)

        ret[..., 0] = temp[..., 2, 1] - temp[..., 1, 2]
        ret[..., 1] = temp[..., 0, 2] - temp[..., 2, 0]
        ret[..., 2] = temp[..., 1, 0] - temp[..., 0, 1]

        return ret

    @staticmethod
    def _data_check_impl(matrix: torch.Tensor) -> bool:
        with torch.no_grad():
            if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
                raise ValueError("3D rotations can only be 3x3 matrices.")

            MATRIX_EPS = theseus.constants._SO3_MATRIX_EPS[matrix.dtype]
            if matrix.dtype != torch.float64:
                matrix = matrix.double()

            _check = (
                torch.matmul(matrix, matrix.transpose(1, 2))
                - torch.eye(3, 3, dtype=matrix.dtype, device=matrix.device)
            ).abs().max().item() < MATRIX_EPS
            _check &= (torch.linalg.det(matrix) - 1).abs().max().item() < MATRIX_EPS

        return _check

    @staticmethod
    def _data_check(matrix: torch.Tensor):
        if not SO3._data_check_impl(matrix):
            raise ValueError("Not valid 3D rotations.")

    @staticmethod
    def _unit_quaternion_check(quaternion: torch.Tensor):
        if quaternion.ndim != 2 or quaternion.shape[1] != 4:
            raise ValueError("Quaternions can only be 4-D vectors.")

        QUANTERNION_EPS = theseus.constants._SO3_QUATERNION_EPS[quaternion.dtype]

        if quaternion.dtype != torch.float64:
            quaternion = quaternion.double()

        if (
            torch.linalg.norm(quaternion, dim=1) - 1
        ).abs().max().item() >= QUANTERNION_EPS:
            raise ValueError("Not unit quaternions.")

    @staticmethod
    def _hat_matrix_check(matrix: torch.Tensor):
        if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
            raise ValueError("Hat matrices of SO(3) can only be 3x3 matrices")

        if (
            matrix.transpose(1, 2) + matrix
        ).abs().max().item() > theseus.constants._SO3_HAT_EPS[matrix.dtype]:
            raise ValueError("Hat matrices of SO(3) can only be skew-symmetric.")

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
        ret.data = (
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
    def normalize(data: torch.Tensor) -> torch.Tensor:
        if data.ndim != 3 or data.shape[1:] != (3, 3):
            raise ValueError("3D rotations can only be 3x3 matrices.")

        U, _, V = torch.svd(data)
        Vtr = V.transpose(1, 2)
        S = torch.diag(torch.tensor([1, 1, -1], dtype=data.dtype, device=data.device))
        temp = (U @ Vtr, U @ S @ Vtr)
        sign = torch.det(temp[0]).reshape([-1, 1, 1]) > 0

        return torch.where(sign, temp[0], temp[1])

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        sine_axis = torch.zeros(self.shape[0], 3, dtype=self.dtype, device=self.device)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        cosine = 0.5 * (self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2] - 1)
        sine = sine_axis.norm(dim=1)
        theta = torch.atan2(sine, cosine)

        near_zero = theta < self._NEAR_ZERO_EPS

        not_near_pi = 1 + cosine > self._NEAR_PI_EPS
        # theta != pi
        near_zero_not_near_pi = near_zero[not_near_pi]
        # Compute the approximation of theta / sin(theta) when theta is near to 0
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)
        sine_nz = torch.where(near_zero_not_near_pi, non_zero, sine[not_near_pi])
        scale = torch.where(
            near_zero_not_near_pi,
            1 + sine[not_near_pi] ** 2 / 6,
            theta[not_near_pi] / sine_nz,
        )
        ret = torch.zeros_like(sine_axis)
        ret[not_near_pi] = sine_axis[not_near_pi] * scale.view(-1, 1)
        # # theta ~ pi
        near_pi = ~not_near_pi
        ddiag = torch.diagonal(self[near_pi], dim1=1, dim2=2)
        # Find the index of major coloumns and diagonals
        major = torch.logical_and(
            ddiag[:, 1] > ddiag[:, 0], ddiag[:, 1] > ddiag[:, 2]
        ) + 2 * torch.logical_and(ddiag[:, 2] > ddiag[:, 0], ddiag[:, 2] > ddiag[:, 1])
        sel_rows = 0.5 * (self[near_pi, major] + self[near_pi, :, major])
        aux = torch.ones(sel_rows.shape[0], dtype=torch.bool)
        sel_rows[aux, major] -= cosine[near_pi]
        axis = sel_rows / sel_rows.norm(dim=1, keepdim=True)
        sign_tmp = sine_axis[near_pi, major].sign()
        sign = torch.where(sign_tmp != 0, sign_tmp, torch.ones_like(sign_tmp))
        ret[near_pi] = axis * (theta[near_pi] * sign).view(-1, 1)

        if jacobians is not None:
            SO3._check_jacobians_list(jacobians)
            jac = torch.zeros_like(self.data)

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
        ret.data = self.data @ so3_2.data
        return ret

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO3":
        # if self.data is a valid SO(3), then self.data.transpose(1, 2) must be valid as well
        return SO3(data=self.data.transpose(1, 2).clone(), strict=False)

    def to_matrix(self) -> torch.Tensor:
        return self.data.clone()

    def to_quaternion(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 4, dtype=self.dtype, device=self.device)

        sine_axis = torch.zeros(self.shape[0], 3, dtype=self.dtype, device=self.device)
        sine_axis[:, 0] = 0.5 * (self[:, 2, 1] - self[:, 1, 2])
        sine_axis[:, 1] = 0.5 * (self[:, 0, 2] - self[:, 2, 0])
        sine_axis[:, 2] = 0.5 * (self[:, 1, 0] - self[:, 0, 1])
        w = 0.5 * (1 + self[:, 0, 0] + self[:, 1, 1] + self[:, 2, 2]).clamp(0, 4).sqrt()

        ret[:, 0] = w

        # theta != pi
        not_near_pi = ret[:, 0] > self._NEAR_PI_EPS
        ret[:, 1:] = 0.5 * sine_axis / w.view(-1, 1)

        # theta ~ pi
        near_pi = ~not_near_pi
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
            * sine_axis[near_pi, major].sign().view(-1, 1)
        )
        sine_half_theta = (0.5 * (1 - cosine_near_pi)).clamp(0, 1).sqrt().view(-1, 1)
        ret[near_pi, 1:] = axis * sine_half_theta

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
        # if self.data is a valid SO(3), so is the copy
        return SO3(data=self.data.clone(), name=new_name, strict=False)

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


rand_so3 = SO3.rand
randn_so3 = SO3.randn
