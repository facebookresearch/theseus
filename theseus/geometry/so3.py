# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, cast

import torch

import theseus.constants

from .lie_group import LieGroup


class SO3(LieGroup):
    def __init__(
        self,
        quaternion: Optional[torch.Tensor] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        raise NotImplementedError

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        raise NotImplementedError

    def dof(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"SO3(data={self.data}, name={self.name})"

    def __str__(self) -> str:
        with torch.no_grad():
            raise NotImplementedError

    def _adjoint_impl(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> LieGroup:
        raise NotImplementedError

    def _log_map_impl(self) -> torch.Tensor:
        raise NotImplementedError

    def _compose_impl(self, so3_2: LieGroup) -> "SO3":
        raise NotImplementedError

    def _inverse_impl(self, get_jacobian: bool = False) -> "SO3":
        raise NotImplementedError

    def to_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def hat(tangent_vector: torch.Tensor) -> torch.Tensor:
        _check = tangent_vector.ndim == 3 and tangent_vector.shape[1:] == (3, 1)
        _check |= tangent_vector.ndim == 2 and tangent_vector.shape[1] == 3
        matrix = torch.zeros(tangent_vector.shape[0], 3, 3).to(
            dtype=tangent_vector.dtype,
            device=tangent_vector.device
        )
        matrix[:, 0, 1] = -tangent_vector[: , 2].view(-1)
        matrix[:, 0, 2] = tangent_vector[: , 1].view(-1)
        matrix[:, 1, 2] = -tangent_vector[: , 0].view(-1)
        matrix[:, 1, 0] = tangent_vector[: , 2].view(-1)
        matrix[:, 2, 0] = -tangent_vector[: , 1].view(-1)
        matrix[:, 2, 1] = tangent_vector[: , 0].view(-1)
        return matrix

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        _check = matrix.ndim == 3 and matrix.shape[1:] == (3, 3)
        _check &= (matrix.transpose(1, 2) + matrix).abs().max().item() < theseus.constants.EPS
        if not _check:
            raise ValueError("Invalid hat matrix for SO3.")
        vec = torch.zeros(matrix.shape[0], 3)
        vec[:, 0] = matrix[:, 2, 1]
        vec[:, 1] = matrix[:, 0, 2]
        vec[:, 2] = matrix[:, 1, 0]

        return vec

    def _copy_impl(self, new_name: Optional[str] = None) -> "SO3":
        return SO3(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SO3":
        return cast(SO3, super().copy(new_name=new_name))
