# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast
from xmlrpc.client import Boolean

import torch

from theseus.geometry.manifold import Manifold

from .lie_group import LieGroup


class Vector(LieGroup):
    def __init__(
        self,
        dof: Optional[int] = None,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if dof is not None and data is not None:
            raise ValueError("Please provide only one of dof or data.")
        if dof is None:
            if data is not None and data.ndim == 1:
                data = data.view(1, -1)
            if data is None or data.ndim != 2:
                raise ValueError(
                    "If dof not provided, data must "
                    "be a tensor of shape (batch_size, dof)"
                )
            dof = data.shape[1]

        super().__init__(dof, data=data, name=name, dtype=dtype)

    # Vector variables are of shape [batch_size, dof]
    @staticmethod
    def _init_data(dof: int) -> torch.Tensor:  # type: ignore
        return torch.zeros(1, dof)

    def dof(self) -> int:
        return self.data.shape[1]

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Boolean = False,
    ) -> "Vector":
        if len(size) != 2:
            raise ValueError("The size should be 2D.")
        return Vector(
            data=2
            * torch.rand(
                size,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
            - 1
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(dof={self.data.shape[1]}, data={self.data}, name={self.name})"
        )

    def allclose(self, other: "Vector", *args, **kwargs) -> bool:
        return torch.allclose(self.data, other.data, *args, **kwargs)

    def __add__(self, other: "Vector") -> "Vector":
        return self.__class__._compose_impl(self, other)

    def __sub__(self, other: "Vector") -> "Vector":
        return self.__class__(data=torch.sub(self.data, other.data))

    def __neg__(self) -> "Vector":
        return self.__class__._inverse_impl(self)

    def __mul__(self, other: Union["Vector", torch.Tensor]) -> "Vector":
        if isinstance(other, Vector):
            return self.__class__(data=torch.mul(self.data, other.data))
        elif isinstance(other, torch.Tensor):
            return self.__class__(data=torch.mul(self.data, other))
        else:
            raise ValueError(
                f"expected type 'Vector' or 'Tensor', got {type(other).__name__}"
            )

    __rmul__ = __mul__

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        if not isinstance(other, torch.Tensor):
            raise ValueError("Vector matmul only accepts torch tensors.")
        if other.ndim != 3:
            raise ValueError(
                f"Vector matmul only accepts tensors with ndim=3 "
                f"but tensor has ndim={other.ndim}."
            )
        return torch.bmm(self.data.unsqueeze(1), other.data).squeeze(1)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        if isinstance(other, Vector):
            raise ValueError("Vector matmul only accepts torch tensors.")
        if other.ndim != 3:
            raise ValueError(
                f"Vector matmul only accepts tensors with ndim=3 but "
                f"tensor has ndim={other.ndim}."
            )
        return torch.bmm(other.data, self.data.unsqueeze(2)).squeeze(2)

    def __truediv__(self, other: Union["Vector", torch.Tensor]) -> "Vector":
        if isinstance(other, Vector):
            return self.__class__(data=torch.div(self.data, other.data))
        elif isinstance(other, torch.Tensor):
            return self.__class__(data=torch.div(self.data, other))
        else:
            raise ValueError(
                f"expected type 'Vector' or 'Tensor', got {type(other).__name__}"
            )

    def dot(self, other: "Vector") -> torch.Tensor:
        return torch.mul(self.data, other.data).sum(-1)

    inner = dot

    def abs(self) -> "Vector":
        return self.__class__(data=torch.abs(self.data))

    def outer(self, other: "Vector") -> torch.Tensor:
        return torch.matmul(self.data.unsqueeze(2), other.data.unsqueeze(1))

    def norm(self, *args, **kwargs) -> torch.Tensor:
        return torch.norm(self.data, *args, **kwargs)

    def cat(self, vecs: Union["Vector", Tuple["Vector"], List["Vector"]]) -> "Vector":
        if isinstance(vecs, Vector):
            result = torch.cat([self.data, vecs.data], 1)
        else:
            result = torch.cat([self.data] + [vec.data for vec in vecs], 1)
        return Vector(data=result)

    def _local_impl(self, vec2: Manifold) -> torch.Tensor:
        if not isinstance(vec2, Vector):
            raise ValueError("Non-vector inputs for Vector.local()")
        else:
            return vec2.data - self.data

    def _local_jacobian(self, _: Manifold) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = torch.eye(self.dof(), dtype=self.dtype, device=self.device).repeat(
            self.shape[0], 1, 1
        )
        return -identity, identity

    def _retract_impl(self, delta: torch.Tensor) -> "Vector":
        return self.__class__(data=torch.add(self.data, delta))

    def _compose_impl(self, vec2: LieGroup) -> "Vector":
        return self.__class__(data=torch.add(self.data, vec2.data))

    def _compose_jacobian(self, _: LieGroup) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = torch.eye(self.dof(), dtype=self.dtype, device=self.device).repeat(
            self.shape[0], 1, 1
        )
        return (identity, identity.clone())

    def _inverse_impl(self) -> "Vector":
        return self.__class__(data=-self.data)

    def _adjoint_impl(self) -> torch.Tensor:
        return (
            torch.eye(self.dof(), dtype=self.dtype, device=self.device)
            .repeat(self.shape[0], 1, 1)
            .to(self.device)
        )

    def _project_impl(self, euclidean_grad: torch.Tensor) -> torch.Tensor:
        self._project_check(euclidean_grad)
        return euclidean_grad.clone()

    @staticmethod
    def exp_map(tangent_vector: torch.Tensor) -> "Vector":
        return Vector(data=tangent_vector.clone())

    def _log_map_impl(self) -> torch.Tensor:
        return self.data.clone()

    def __hash__(self):
        return id(self)

    def _copy_impl(self, new_name: Optional[str] = None) -> "Vector":
        return self.__class__(data=self.data.clone(), name=new_name)

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "Vector":
        return cast(Vector, super().copy(new_name=new_name))
