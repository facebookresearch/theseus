# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, cast

import torch

import theseus.constants

from .lie_group import LieGroup
from .point_types import Point2
from .so2 import SO2


# If data is passed, must be x, y, cos, sin
# If x_y_theta is passed, must be tensor with shape batch_size x 3
class SE2(LieGroup):
    SE2_EPS = 5e-7

    def __init__(
        self,
        x_y_theta: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if x_y_theta is not None and data is not None:
            raise ValueError("Please provide only one of x_y_theta or data.")
        if x_y_theta is not None:
            dtype = x_y_theta.dtype
        super().__init__(data=data, name=name, dtype=dtype)
        if x_y_theta is not None:
            rotation = SO2(theta=x_y_theta[:, 2:])
            translation = Point2(data=x_y_theta[:, :2])
            self.update_from_rot_and_trans(rotation, translation)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SE2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        x_y_theta = torch.rand(
            size[0],
            3,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        x_y_theta[:, 2] = 2 * theseus.constants.PI * (x_y_theta[:, 2] - 0.5)

        return SE2(x_y_theta=x_y_theta)

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SE2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        x_y_theta = torch.randn(
            size[0],
            3,
            generator=generator,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        x_y_theta[:, 2] *= theseus.constants.PI

        return SE2(x_y_theta=x_y_theta)

    def _transform_shape_check(self, point: Union[Point2, torch.Tensor]):
        err_msg = (
            f"SE2 can only transform vectors of shape [{self.shape[0]}, 2] or [1, 2], "
            f"but the input has shape {point.shape}."
        )

        if isinstance(point, torch.Tensor):
            if not point.ndim == 2 or point.shape[1] != 2:
                raise ValueError(err_msg)
        elif point.dof() != 2:
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
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.tensor([0.0, 0.0, 1.0, 0.0]).view(1, 4)

    def dof(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"SE2(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            theta = torch.atan2(self[:, 3:], self[:, 2:3])
            xytheta = torch.cat([self[:, :2], theta], dim=1)
            return f"SE2(xytheta={xytheta}, name={self.name})"

    @property
    def rotation(self) -> SO2:
        return SO2(data=self[:, 2:])

    def theta(self, jacobians: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            J_out = torch.zeros(
                self.shape[0], 1, 3, device=self.device, dtype=self.dtype
            )
            J_out[:, 0, 2] = 1
            jacobians.append(J_out)
        return self.rotation.theta()

    @property
    def translation(self) -> Point2:
        return Point2(data=self[:, :2])

    def xy(self, jacobians: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            rotation = self.rotation
            J_out = torch.zeros(
                self.shape[0], 2, 3, device=self.device, dtype=self.dtype
            )
            J_out[:, :2, :2] = rotation.to_matrix()
            jacobians.append(J_out)
        return self.translation.data

    def update_from_rot_and_trans(self, rotation: SO2, translation: Point2):
        batch_size = rotation.shape[0]
        self.data = torch.empty(batch_size, 4).to(
            device=rotation.device, dtype=rotation.dtype
        )
        self[:, :2] = translation.data
        cosine, sine = rotation.to_cos_sin()
        self[:, 2] = cosine
        self[:, 3] = sine

    def _log_map_impl(self) -> torch.Tensor:
        rotation = self.rotation
        theta = rotation.log_map().view(-1)
        cosine, sine = rotation.to_cos_sin()

        # Compute the approximations when theta is near to 0
        small_theta = theta.abs() < SE2.SE2_EPS
        non_zero = torch.ones(1, dtype=self.dtype, device=self.device)
        sine_nz = torch.where(small_theta, non_zero, sine)
        half_theta_by_tan_half_theta = (
            0.5
            * (1 + cosine)
            * torch.where(small_theta, 1 + sine**2 / 6, theta / sine_nz)
        )
        half_theta = 0.5 * theta

        # Compute the translation
        ux = half_theta_by_tan_half_theta * self[:, 0] + half_theta * self[:, 1]
        uy = half_theta_by_tan_half_theta * self[:, 1] - half_theta * self[:, 0]

        return torch.stack((ux, uy, theta), dim=1)

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> "SE2":
        u = tangent_vector[:, :2]
        theta = tangent_vector[:, 2]
        rotation = SO2(theta=theta)

        cosine, sine = rotation.to_cos_sin()

        # Compute the approximations when theta is near to 0
        small_theta = theta.abs() < SE2.SE2_EPS
        non_zero = torch.ones(
            1, dtype=tangent_vector.dtype, device=tangent_vector.device
        )
        theta_nz = torch.where(small_theta, non_zero, theta)
        sine_by_theta = torch.where(small_theta, 1 - theta**2 / 6, sine / theta_nz)
        cosine_minus_one_by_theta = torch.where(
            small_theta, -theta / 2 + theta**3 / 24, (cosine - 1) / theta_nz
        )

        # Compute the translation
        x = sine_by_theta * u[:, 0] + cosine_minus_one_by_theta * u[:, 1]
        y = sine_by_theta * u[:, 1] - cosine_minus_one_by_theta * u[:, 0]
        translation = Point2(data=torch.stack((x, y), dim=1))

        se2 = SE2(dtype=tangent_vector.dtype)
        se2.update_from_rot_and_trans(rotation, translation)
        return se2

    def _adjoint_impl(self) -> torch.Tensor:
        ret = torch.zeros(self.shape[0], 3, 3).to(device=self.device, dtype=self.dtype)
        ret[:, :2, :2] = self.rotation.to_matrix()
        ret[:, 0, 2] = self[:, 1]
        ret[:, 1, 2] = -self[:, 0]
        ret[:, 2, 0] = 0
        ret[:, 2, 2] = 1
        return ret

    def _compose_impl(self, se2_2: LieGroup) -> "SE2":
        se2_2 = cast(SE2, se2_2)
        rotation_1 = self.rotation
        rotation_2 = se2_2.rotation
        translation_1 = cast(Point2, self.translation)
        translation_2 = cast(Point2, se2_2.translation)
        new_rotation = rotation_1.compose(rotation_2)
        new_translation = cast(
            Point2,
            translation_1.compose(rotation_1.rotate(translation_2)),
        )
        return SE2(data=torch.cat([new_translation.data, new_rotation.data], dim=1))

    def _inverse_impl(self) -> "SE2":
        inverse_rotation = self.rotation._inverse_impl()
        inverse_translation = inverse_rotation.rotate(cast(Point2, -self.translation))
        se2_inverse = SE2(dtype=self.dtype)
        se2_inverse.update_from_rot_and_trans(inverse_rotation, inverse_translation)
        return se2_inverse

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)
        ret = torch.zeros(
            euclidean_grad.shape[:-1] + torch.Size([3]),
            dtype=self.dtype,
            device=self.device,
        )

        temp = torch.stack((-self[:, 3], self[:, 2]), dim=1)

        if is_sparse:
            ret[..., 0] = torch.einsum(
                "i...k,i...k->i...", euclidean_grad[..., :2], self[:, 2:]
            )
            ret[..., 1] = torch.einsum(
                "i...k,i...k->i...", euclidean_grad[..., :2], temp
            )
            ret[..., 2] = torch.einsum(
                "i...k,i...k->i...", euclidean_grad[..., 2:], temp
            )
        else:
            ret[..., 0] = torch.einsum(
                "...k,...k", euclidean_grad[..., :2], self[:, 2:]
            )
            ret[..., 1] = torch.einsum("...k,...k", euclidean_grad[..., :2], temp)
            ret[..., 2] = torch.einsum("...k,...k", euclidean_grad[..., 2:], temp)
        return ret

    def to_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(self.shape[0], 3, 3).to(
            device=self.device, dtype=self.dtype
        )
        matrix[:, :2, :2] = self.rotation.to_matrix()
        matrix[:, :2, 2] = self[:, :2]
        matrix[:, 2, 2] = 1.0
        return matrix

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        theta = tangent_vector[:, 2]
        u = tangent_vector[:, :2]
        matrix = torch.zeros(tangent_vector.shape[0], 3, 3).to(
            device=tangent_vector.device, dtype=tangent_vector.dtype
        )
        matrix[:, 0, 1] = -theta
        matrix[:, 1, 0] = theta
        matrix[:, :2, 2] = u
        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        _check = matrix.ndim == 3 and matrix.shape[1:] == (3, 3)
        _check &= torch.allclose(matrix[:, 0, 1], -matrix[:, 1, 0])
        _check &= matrix[:, 0, 0].abs().max().item() < theseus.constants.EPS
        _check &= matrix[:, 1, 1].abs().max().item() < theseus.constants.EPS
        _check &= matrix[:, 2, 2].abs().max().item() < theseus.constants.EPS
        if not _check:
            raise ValueError("Invalid hat matrix for SE2.")
        batch_size = matrix.shape[0]
        tangent_vector = torch.zeros(batch_size, 3).to(
            device=matrix.device, dtype=matrix.dtype
        )
        tangent_vector[:, 2] = matrix[:, 1, 0]
        tangent_vector[:, :2] = matrix[:, :2, 2]
        return tangent_vector

    def transform_to(
        self,
        point: Union[torch.Tensor, Point2],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point2:
        self._transform_shape_check(point)
        batch_size = max(self.shape[0], point.shape[0])
        if isinstance(point, torch.Tensor):
            p = point
        else:
            p = point.data

        cosine = self[:, 2]
        sine = self[:, 3]
        temp = p - self[:, :2]
        ret = SO2._rotate_from_cos_sin(temp, cosine, -sine)

        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            Jg = torch.zeros(batch_size, 2, 3, dtype=self.dtype, device=self.device)
            Jg[:, 0, 0] = -1
            Jg[:, 1, 1] = -1
            Jg[:, 0, 2] = ret.y()
            Jg[:, 1, 2] = -ret.x()

            Jpnt = torch.zeros(batch_size, 2, 2, dtype=self.dtype, device=self.device)
            Jpnt[:, 0, 0] = cosine
            Jpnt[:, 0, 1] = sine
            Jpnt[:, 1, 0] = -sine
            Jpnt[:, 1, 1] = cosine

            jacobians.extend([Jg, Jpnt])

        return ret

    def transform_from(
        self,
        point: Union[torch.Tensor, Point2],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point2:
        self._transform_shape_check(point)
        batch_size = max(self.shape[0], point.shape[0])

        cosine = self[:, 2]
        sine = self[:, 3]
        temp = SO2._rotate_from_cos_sin(point, cosine, sine)
        ret = Point2(data=temp.data + self[:, :2])

        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            Jg = torch.zeros(batch_size, 2, 3, dtype=self.dtype, device=self.device)
            Jg[:, 0, 0] = cosine
            Jg[:, 0, 1] = -sine
            Jg[:, 1, 0] = sine
            Jg[:, 1, 1] = cosine
            Jg[:, 0, 2] = -temp.y()
            Jg[:, 1, 2] = temp.x()

            Jpnt = torch.zeros(batch_size, 2, 2, dtype=self.dtype, device=self.device)
            Jpnt[:, 0, 0] = cosine
            Jpnt[:, 0, 1] = -sine
            Jpnt[:, 1, 0] = sine
            Jpnt[:, 1, 1] = cosine

            jacobians.extend([Jg, Jpnt])

        return ret

    def _copy_impl(self, new_name: Optional[str] = None) -> "SE2":
        return SE2(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SE2":
        return cast(SE2, super().copy(new_name=new_name))


rand_se2 = SE2.rand
randn_se2 = SE2.randn
