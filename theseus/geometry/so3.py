# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, cast

import torch

from .lie_group import LieGroup


class SO3(LieGroup):
    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        raise NotImplementedError

    @staticmethod
    def _init_data() -> torch.Tensor:  # type: ignore
        raise NotImplementedError

    def dof(self) -> int:
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def vee(matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _copy_impl(self, new_name: Optional[str] = None) -> "SO3":
        return SO3(data=self.data.clone(), name=new_name)

    # only added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "SO3":
        return cast(SO3, super().copy(new_name=new_name))
