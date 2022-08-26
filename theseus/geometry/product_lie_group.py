# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Any, cast, Type

import torch

from .lie_group import LieGroup


def ProductLieGroup(groups: List[LieGroup]):
    class _ProductLieGroup(LieGroup):
        _group_clses: List[Type[LieGroup]] = [type(group) for group in groups]
        _shapes: List[List[int]] = [list(group.shape[1:]) for group in groups]
        _dofs: List[int] = [0]
        _numels: List[int] = [0]

        for group in groups:
            _dofs.append(_dofs[-1] + group.dof())
            _numels.append(_numels[-1] + group.numel())

        def __init__(
            self,
            groups: List[LieGroup],
            name: Optional[str] = None,
            dtype: Optional[torch.dtype] = None,
            strict: bool = False,
            new_copy: bool = True,
            group_cls_check: bool = True,
        ):
            if group_cls_check:
                if not all(
                    [
                        group.__class__ == group_cls
                        for group, group_cls in zip(groups, self._group_clses)
                    ]
                ):
                    raise ValueError(
                        "All the groups must be the instances of the given Lie group classes."
                    )

            self.groups: List[LieGroup] = (
                List([group.copy() for group in groups]) if new_copy else groups
            )
            tensor = torch.cat(
                [group.tensor.view(group.shape[0], -1) for group in self.groups], dim=-1
            )
            super().__init__(True, tensor=tensor, name=name, dtype=dtype, strict=strict)

        @staticmethod
        def rand(
            *size: int,
            generator: Optional[torch.Generator] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = False,
        ) -> "LieGroup":
            return _ProductLieGroup(
                groups=[
                    cast(LieGroup, group_cls).rand(
                        *size,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                        requires_grad=requires_grad,
                    )
                    for group_cls in _ProductLieGroup._group_clses
                ],
                new_copy=False,
                group_cls_check=False,
            )

        @staticmethod
        def randn(
            *size: int,
            generator: Optional[torch.Generator] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            requires_grad: bool = False,
        ) -> "LieGroup":
            return _ProductLieGroup(
                groups=[
                    cast(LieGroup, group_cls).randn(
                        *size,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                        requires_grad=requires_grad,
                    )
                    for group_cls in _ProductLieGroup._group_clses
                ],
                new_copy=False,
                group_cls_check=False,
            )

        @staticmethod
        def _init_tensor(*args: Any) -> torch.Tensor:
            return torch.empty([1, _ProductLieGroup._numels[-1]])

        def dof(self) -> int:
            return self._dofs[-1]

        def numel(self) -> int:
            return self._numels[-1]

        @staticmethod
        def exp_map(
            tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
        ) -> "LieGroup":
            if (
                tangent_vector.ndim != 2
                or tangent_vector.shape[1] != _ProductLieGroup._dofs[-1]
            ):
                raise ValueError(
                    f"Tangent vectors should be {_ProductLieGroup._dofs[-1]}-D vectors."
                )
            groups = [
                cast(LieGroup, group_cls).exp_map(
                    tangent_vector[
                        :, _ProductLieGroup._dofs[i] : _ProductLieGroup._dofs[i + 1]
                    ]
                )
                for i, group_cls in enumerate(_ProductLieGroup._group_clses)
            ]
            return _ProductLieGroup(
                groups=groups, new_copy=False, group_cls_check=False
            )

        def _retract_impl(self, delta: torch.Tensor) -> "LieGroup":
            groups_plus = cast(
                List[LieGroup],
                [
                    group.retract(delta[:, self._dofs[i] : self._dofs[i + 1]])
                    for i, group in enumerate(self.groups)
                ],
            )
            return _ProductLieGroup(
                groups=groups_plus, new_copy=False, group_cls_check=False
            )

        def _adjoint_impl(self) -> torch.Tensor:
            adjoint = self.tensor.new_zeros([self.shape[0], self.dof(), self.dof()])
            for i, group in enumerate(self.groups):
                adjoint[
                    :,
                    self._dofs[i] : self._dofs[i + 1],
                    self._dofs[i] : self._dofs[i + 1],
                ] = group.adjoint()

            return adjoint

        def to_matrix(self) -> torch.Tensor:
            return self.tensor

        def _compose_impl(self, variable2: "LieGroup") -> "_ProductLieGroup":
            return _ProductLieGroup(
                groups=[
                    group1.compose(group2)
                    for group1, group2 in zip(
                        self.groups, cast(_ProductLieGroup, variable2).groups
                    )
                ],
                new_copy=False,
                group_cls_check=False,
            )

        def _copy_impl(self, new_name: Optional[str] = None) -> "_ProductLieGroup":
            return _ProductLieGroup(
                groups=[group.copy() for group in self.groups],
                new_copy=False,
                group_cls_check=False,
            )

        def _log_map_impl(
            self, jacobians: Optional[List[torch.Tensor]] = None
        ) -> torch.Tensor:
            return torch.cat([group.log_map() for group in self.groups], dim=-1)

        def _inverse_impl(self) -> "_ProductLieGroup":
            groups_inv = [group.inverse() for group in self.groups]
            return _ProductLieGroup(
                groups=groups_inv, new_copy=False, group_cls_check=False
            )

        def _project_impl(
            self, euclidean_grad: torch.Tensor, is_sparse: bool = False
        ) -> torch.Tensor:
            raise NotImplementedError

        @staticmethod
        def _check_tensor_impl(tensor: torch.Tensor) -> bool:
            return all(
                [
                    cast(LieGroup, group_cls)._check_tensor_impl(
                        tensor[
                            :,
                            _ProductLieGroup._numels[i] : _ProductLieGroup._numels[
                                i + 1
                            ],
                        ].view([-1] + _ProductLieGroup._shapes[i])
                    )
                    for i, group_cls in enumerate(_ProductLieGroup._group_clses)
                ]
            )

        @staticmethod
        def normalize(tensor: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    cast(LieGroup, group_cls)
                    .normalize(
                        tensor[
                            :,
                            _ProductLieGroup._numels[i] : _ProductLieGroup._numels[
                                i + 1
                            ],
                        ].view([-1] + _ProductLieGroup._shapes[i])
                    )
                    .view(tensor.shape[0], -1)
                    for i, group_cls in enumerate(_ProductLieGroup._group_clses)
                ],
                dim=-1,
            )

        @staticmethod
        def group_clses() -> List[Type[LieGroup]]:
            return _ProductLieGroup._group_clses

    return _ProductLieGroup
