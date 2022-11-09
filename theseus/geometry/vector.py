# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

import torch

from theseus.constants import DeviceType
from theseus.geometry.manifold import Manifold

from .lie_group import LieGroup


class Vector(LieGroup):
    def __init__(
        self,
        dof: Optional[int] = None,
        tensor: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if dof is not None and tensor is not None:
            raise ValueError("Please provide only one of dof or tensor.")
        if dof is None:
            if tensor is not None and tensor.ndim == 1:
                tensor = tensor.view(1, -1)
            if tensor is None or tensor.ndim != 2:
                raise ValueError(
                    "If dof not provided, tensor must " "have shape (batch_size, dof)"
                )
            dof = tensor.shape[1]

        super().__init__(dof, tensor=tensor, name=name, dtype=dtype)

    # Vector variables are of shape [batch_size, dof]
    @staticmethod
    def _init_tensor(dof: int) -> torch.Tensor:  # type: ignore
        return torch.zeros(1, dof)

    def dof(self) -> int:
        return self.tensor.shape[1]

    @staticmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "Vector":
        if len(size) != 2:
            raise ValueError("The size should be 2D.")
        return Vector(
            tensor=torch.rand(
                size,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    @staticmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> "Vector":
        if len(size) != 2:
            raise ValueError("The size should be 2D.")
        return Vector(
            tensor=torch.randn(
                size,
                generator=generator,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(dof={self.tensor.shape[1]}, tensor={self.tensor}, name={self.name})"
        )

    def allclose(self, other: "Vector", *args, **kwargs) -> bool:
        return torch.allclose(self.tensor, other.tensor, *args, **kwargs)

    def __add__(self, other: "Vector") -> "Vector":
        return self.__class__._compose_impl(self, other)

    def __sub__(self, other: "Vector") -> "Vector":
        return self.__class__(tensor=torch.sub(self.tensor, other.tensor))

    def __neg__(self) -> "Vector":
        return self.__class__._inverse_impl(self)

    def __mul__(self, other: Union["Vector", torch.Tensor]) -> "Vector":
        if isinstance(other, Vector):
            return self.__class__(tensor=torch.mul(self.tensor, other.tensor))
        elif isinstance(other, torch.Tensor):
            return self.__class__(tensor=torch.mul(self.tensor, other))
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
        return torch.bmm(self.tensor.unsqueeze(1), other).squeeze(1)

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        if isinstance(other, Vector):
            raise ValueError("Vector matmul only accepts torch tensors.")
        if other.ndim != 3:
            raise ValueError(
                f"Vector matmul only accepts tensors with ndim=3 but "
                f"tensor has ndim={other.ndim}."
            )
        return torch.bmm(other, self.tensor.unsqueeze(2)).squeeze(2)

    def __truediv__(self, other: Union["Vector", torch.Tensor]) -> "Vector":
        if isinstance(other, Vector):
            return self.__class__(tensor=torch.div(self.tensor, other.tensor))
        elif isinstance(other, torch.Tensor):
            return self.__class__(tensor=torch.div(self.tensor, other))
        else:
            raise ValueError(
                f"expected type 'Vector' or 'Tensor', got {type(other).__name__}"
            )

    def dot(self, other: "Vector") -> torch.Tensor:
        return torch.mul(self.tensor, other.tensor).sum(-1)

    inner = dot

    def abs(self) -> "Vector":
        return self.__class__(tensor=torch.abs(self.tensor))

    def outer(self, other: "Vector") -> torch.Tensor:
        return torch.matmul(self.tensor.unsqueeze(2), other.tensor.unsqueeze(1))

    def norm(self, *args, **kwargs) -> torch.Tensor:
        return torch.norm(self.tensor, *args, **kwargs)

    def cat(self, vecs: Union["Vector", Tuple["Vector"], List["Vector"]]) -> "Vector":
        if isinstance(vecs, Vector):
            result = torch.cat([self.tensor, vecs.tensor], 1)
        else:
            result = torch.cat([self.tensor] + [vec.tensor for vec in vecs], 1)
        return Vector(tensor=result)

    def to_matrix(self) -> torch.Tensor:
        return self.tensor.clone()

    def _local_impl(
        self, vec2: Manifold, jacobians: List[torch.Tensor] = None
    ) -> torch.Tensor:
        if not isinstance(vec2, Vector):
            raise ValueError("Non-vector inputs for Vector.local()")
        else:
            return LieGroup._local_impl(self, vec2, jacobians)

    def _retract_impl(self, delta: torch.Tensor) -> "Vector":
        return self.__class__(tensor=torch.add(self.tensor, delta))

    def _compose_impl(self, vec2: LieGroup) -> "Vector":
        return self.__class__(tensor=torch.add(self.tensor, vec2.tensor))

    def _compose_jacobian(self, _: LieGroup) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = torch.eye(self.dof(), dtype=self.dtype, device=self.device).repeat(
            self.shape[0], 1, 1
        )
        return (identity, identity.clone())

    def _inverse_impl(self) -> "Vector":
        return self.__class__(tensor=-self.tensor)

    def _adjoint_impl(self) -> torch.Tensor:
        return (
            torch.eye(self.dof(), dtype=self.dtype, device=self.device)
            .repeat(self.shape[0], 1, 1)
            .to(self.device)
        )

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        self._project_check(euclidean_grad, is_sparse)
        return euclidean_grad.clone()

    @staticmethod
    def exp_map(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
    ) -> "Vector":
        if tangent_vector.ndim != 2:
            raise ValueError("The dimension of tangent vectors should be 2.")

        Vector._exp_map_jacobian_impl(tangent_vector, jacobians)

        return Vector(tensor=tangent_vector.clone())

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2:
            raise ValueError("Vector variables expect tensors with ndim=2.")

        return True

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError("Vector variables expect tensors with ndim=2.")

        return tensor

    @staticmethod
    def _exp_map_jacobian_impl(
        tangent_vector: torch.Tensor, jacobians: Optional[List[torch.Tensor]]
    ):
        if jacobians is not None:
            shape = tangent_vector.shape
            LieGroup._check_jacobians_list(jacobians)
            jacobians.append(
                torch.eye(
                    shape[1], dtype=tangent_vector.dtype, device=tangent_vector.device
                ).repeat(shape[0], 1, 1)
            )

    def _log_map_impl(
        self, jacobians: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        self._log_map_jacobian_impl(jacobians)
        return self.tensor.clone()

    def _log_map_jacobian_impl(self, jacobians: Optional[List[torch.Tensor]] = None):
        if jacobians is not None:
            shape = self.shape
            Vector._check_jacobians_list(jacobians)
            jacobians.append(
                torch.eye(shape[1], dtype=self.dtype, device=self.device).repeat(
                    shape[0], 1, 1
                )
            )

    def __hash__(self):
        return id(self)

    def _copy_impl(self, new_name: Optional[str] = None) -> "Vector":
        return self.__class__(tensor=self.tensor.clone(), name=new_name)

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "Vector":
        return cast(Vector, super().copy(new_name=new_name))


rand_vector = Vector.rand
randn_vector = Vector.randn
