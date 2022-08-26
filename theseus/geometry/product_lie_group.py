# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Any, cast

import torch

from .lie_group import LieGroup


class ProductLieGroup(LieGroup):
    def __init__(
        self,
        groups: List[LieGroup],
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        strict: bool = False,
    ):
        self.groups: List[LieGroup] = List([group.copy() for group in groups])
        super().__init__(
            tensor=torch.empty([1, 0]), name=name, dtype=dtype, strict=strict
        )
        self.tensor = torch.cat(
            [group.tensor.view(group.shape[0], -1) for group in self.groups]
        )
        self._dofs: List[int] = [0]
        self._numels: List[int] = [0]
        for group in self.groups:
            self._dofs.append(self._dofs[-1] + group.dof())
            self._numels.append(self._numels[-1] + group.numel())

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "LieGroup":
        raise NotImplementedError

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "LieGroup":
        raise NotImplementedError

    @staticmethod
    def _init_tensor(*args: Any) -> torch.Tensor:
        return torch.empty([1, 0])

    def dof(self) -> int:
        return self._dofs[-1]

    def numel(self) -> int:
        return self._numels[-1]

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "LieGroup":
        raise NotImplementedError

    def _retract_impl(self, delta: torch.Tensor) -> "LieGroup":
        groups_plus = cast(
            List[LieGroup],
            [
                group.retract(delta[:, self._dofs[i] : self._dofs[i + 1]])
                for i, group in enumerate(self.groups)
            ],
        )
        return ProductLieGroup(groups=groups_plus)

    def _adjoint_impl(self) -> torch.Tensor:
        adjoint = self.tensor.new_zeros([self.shape[0], self.dof(), self.dof()])
        for i, group in enumerate(self.groups):
            adjoint[
                :, self._dofs[i] : self._dofs[i + 1], self._dofs[i] : self._dofs[i + 1]
            ] = group.adjoint()

        return adjoint

    def to_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    def _compose_impl(self, variable2: "LieGroup") -> "ProductLieGroup":
        raise NotImplementedError

    def _copy_impl(self, new_name: Optional[str] = None) -> "ProductLieGroup":
        raise NotImplementedError

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def _inverse_impl(self) -> "ProductLieGroup":
        raise NotImplementedError
