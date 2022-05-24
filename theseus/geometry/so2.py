from typing import List, Optional, Tuple, Union, cast

import torch

import theseus.constants

from .lie_group import LieGroup
from .point_types import Point2


class SO2(LieGroup):
    def __init__(
        self,
        theta: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if theta is not None and data is not None:
            raise ValueError("Please provide only one of theta or data.")
        if theta is not None:
            dtype = theta.dtype
        super().__init__(data=data, name=name, dtype=dtype)
        if theta is not None:
            if theta.ndim == 1:
                theta = theta.unsqueeze(1)
            if theta.ndim != 2 or theta.shape[1] != 1:
                raise ValueError(
                    "Argument theta must be have ndim = 1, or ndim=2 and shape[1] = 1."
                )
            self.update_from_angle(theta)

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "SO2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return SO2.exp_map(
            2
            * theseus.constants.PI
            * torch.rand(
                size[0],
                1,
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
    ) -> "SO2":
        if len(size) != 1:
            raise ValueError("The size should be 1D.")
        return SO2.exp_map(
            theseus.constants.PI
            * torch.randn(
                size[0],
                1,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        return torch.tensor([1.0, 0.0]).view(1, 2)

    def update_from_angle(self, theta: torch.Tensor):
        self.update(torch.cat([theta.cos(), theta.sin()], dim=1))

    def dof(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"SO2(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            theta = torch.atan2(self[:, 1:], self[:, 0:1])
            return f"SO2(theta={theta}, name={self.name})"

    def theta(self) -> torch.Tensor:
        return self.log_map()

    def _adjoint_impl(self) -> torch.Tensor:
        return torch.ones(self.shape[0], 1, 1, device=self.device, dtype=self.dtype)

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)

        temp = torch.stack((-self[:, 1], self[:, 0]), dim=1)

        if is_sparse:
            return torch.einsum("i...k,i...k->i...", euclidean_grad, temp).unsqueeze(-1)
        else:
            return torch.einsum("...k,...k", euclidean_grad, temp).unsqueeze(-1)

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "SO2":
        so2 = SO2(dtype=tangent_vector.dtype)
        so2.update_from_angle(tangent_vector)

        if jacobians is not None:
            SO2._check_jacobians_list(jacobians)
            jacobians.append(
                torch.ones(
                    tangent_vector.shape[0],
                    1,
                    1,
                    dtype=tangent_vector.dtype,
                    device=tangent_vector.device,
                )
            )

        return so2

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if jacobians is not None:
            SO2._check_jacobians_list(jacobians)
            jacobians.append(
                torch.ones(
                    self.shape[0],
                    1,
                    1,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        cosine, sine = self.to_cos_sin()
        return torch.atan2(sine, cosine).unsqueeze(1)

    def _compose_impl(self, so2_2: LieGroup) -> "SO2":
        so2_2 = cast(SO2, so2_2)
        cos_1, sin_1 = self.to_cos_sin()
        cos_2, sin_2 = so2_2.to_cos_sin()
        new_cos = cos_1 * cos_2 - sin_1 * sin_2
        new_sin = sin_1 * cos_2 + cos_1 * sin_2
        return SO2(data=torch.stack([new_cos, new_sin], dim=1))

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO2":
        cosine, sine = self.to_cos_sin()
        return SO2(data=torch.stack([cosine, -sine], dim=1))

    def _rotate_shape_check(self, point: Union[Point2, torch.Tensor]):
        err_msg = (
            f"SO2 can only transform vectors of shape [{self.shape[0]}, 2] or [1, 2], "
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
    def _rotate_from_cos_sin(
        point: Union[Point2, torch.Tensor],
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> Point2:
        batch_size = max(point.shape[0], cosine.shape[0])
        if isinstance(point, torch.Tensor):
            if point.ndim != 2 or point.shape[1] != 2:
                raise ValueError(
                    f"Point tensor must have shape batch_size x 2, "
                    f"but received {point.shape}."
                )
            point_data = point
        else:
            point_data = point.data
        px, py = point_data[:, 0], point_data[:, 1]
        new_point_data = torch.empty(
            batch_size, 2, device=cosine.device, dtype=cosine.dtype
        )
        new_point_data[:, 0] = cosine * px - sine * py
        new_point_data[:, 1] = sine * px + cosine * py
        return Point2(data=new_point_data)

    def rotate(
        self,
        point: Union[Point2, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point2:
        self._rotate_shape_check(point)
        cosine, sine = self.to_cos_sin()
        ret = SO2._rotate_from_cos_sin(point, cosine, sine)
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            Jrot = torch.stack([-ret.y(), ret.x()], dim=1).view(-1, 2, 1)
            Jpnt = self.to_matrix().expand(ret.shape[0], -1, -1)
            jacobians.extend([Jrot, Jpnt])
        return ret

    def unrotate(
        self,
        point: Union[Point2, torch.Tensor],
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> Point2:
        self._rotate_shape_check(point)
        cosine, sine = self.to_cos_sin()
        ret = SO2._rotate_from_cos_sin(point, cosine, -sine)
        if jacobians is not None:
            self._check_jacobians_list(jacobians)
            Jrot = torch.stack([ret.y(), -ret.x()], dim=1).view(-1, 2, 1)
            Jpnt = self.to_matrix().transpose(2, 1).expand(ret.shape[0], -1, -1)
            jacobians.extend([Jrot, Jpnt])
        return ret

    def to_cos_sin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self[:, 0], self[:, 1]

    def to_matrix(self) -> torch.Tensor:
        matrix = torch.empty(self.shape[0], 2, 2).to(
            device=self.device, dtype=self.dtype
        )
        cosine, sine = self.to_cos_sin()
        matrix[:, 0, 0] = cosine
        matrix[:, 0, 1] = -sine
        matrix[:, 1, 0] = sine
        matrix[:, 1, 1] = cosine
        return matrix

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        matrix = torch.zeros(tangent_vector.shape[0], 2, 2).to(
            dtype=tangent_vector.dtype,
            device=tangent_vector.device,
        )
        matrix[:, 0, 1] = -tangent_vector.view(-1)
        matrix[:, 1, 0] = tangent_vector.view(-1)
        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        _check = matrix.ndim == 3 and matrix.shape[1:] == (2, 2)
        _check &= matrix[:, 0, 0].abs().max().item() < theseus.constants.EPS
        _check &= matrix[:, 1, 1].abs().max().item() < theseus.constants.EPS
        _check &= torch.allclose(matrix[:, 0, 1], -matrix[:, 1, 0])
        if not _check:
            raise ValueError("Invalid hat matrix for SO2.")
        return matrix[:, 1, 0].clone().view(-1, 1)

    def _copy_impl(self, new_name: Optional[str] = None) -> "SO2":
        return SO2(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SO2":
        return cast(SO2, super().copy(new_name=new_name))


rand_so2 = SO2.rand
randn_so2 = SO2.randn
