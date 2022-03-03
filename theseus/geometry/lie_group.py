# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, List, Optional, Tuple, cast

import torch

from theseus.geometry.manifold import Manifold


# Abstract class to represent Lie groups.
# Concrete classes must implement the following methods:
#   - `exp_map`
#   - `_log_map`
#   - `_adjoint`
#   _ `_compose`
#   _ `_inverse`
#
# Constructor can optionally provide an initial data value as a keyword argument.
class LieGroup(Manifold):
    def __init__(
        self,
        *args: Any,
        data: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__(*args, data=data, name=name, dtype=dtype)

    @staticmethod
    def _check_jacobians_list(jacobians: List[torch.Tensor]):
        if len(jacobians) != 0:
            raise ValueError("jacobians list to be populated must be empty.")

    @staticmethod
    @abc.abstractmethod
    def _init_data(*args: Any) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def dof(self) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "LieGroup":
        pass

    @staticmethod
    @abc.abstractmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> "LieGroup":
        pass

    def __str__(self) -> str:
        return repr(self)

    @staticmethod
    @abc.abstractmethod
    def exp_map(tangent_vector: torch.Tensor) -> "LieGroup":
        pass

    @abc.abstractmethod
    def _log_map_impl(self) -> torch.Tensor:
        pass

    def log_map(self) -> torch.Tensor:
        return self._log_map_impl()

    @abc.abstractmethod
    def _adjoint_impl(self) -> torch.Tensor:
        pass

    def adjoint(self) -> torch.Tensor:
        return self._adjoint_impl()

    def _project_check(self, euclidean_grad: torch.Tensor, is_sparse: bool = False):
        if euclidean_grad.dtype != self.dtype:
            raise ValueError(
                "Euclidean gradients must be of the same type as the Lie group."
            )

        if euclidean_grad.device != self.device:
            raise ValueError(
                "Euclidean gradients must be on the same device as the Lie group."
            )

        if euclidean_grad.shape[-self.ndim + is_sparse :] != self.shape[is_sparse:]:
            raise ValueError(
                "Euclidean gradients must have consistent shapes with the Lie group."
            )

    def between(
        self, variable2: "LieGroup", jacobians: Optional[List[torch.Tensor]] = None
    ) -> "LieGroup":
        v1_inverse = self._inverse_impl()
        between = v1_inverse._compose_impl(variable2)
        if jacobians is not None:
            LieGroup._check_jacobians_list(jacobians)
            Jinv = LieGroup._inverse_jacobian(self)
            Jcmp0, Jcmp1 = v1_inverse._compose_jacobian(variable2)
            Jbetween0 = torch.matmul(Jcmp0, Jinv)
            jacobians.extend([Jbetween0, Jcmp1])
        return between

    @abc.abstractmethod
    def _compose_impl(self, variable2: "LieGroup") -> "LieGroup":
        pass

    def compose(
        self, variable2: "LieGroup", jacobians: Optional[List[torch.Tensor]] = None
    ) -> "LieGroup":
        composition = self._compose_impl(variable2)
        if jacobians is not None:
            LieGroup._check_jacobians_list(jacobians)
            jacobians.extend(self._compose_jacobian(variable2))
        return composition

    @abc.abstractmethod
    def _inverse_impl(self) -> "LieGroup":
        pass

    def inverse(self, jacobian: Optional[List[torch.Tensor]] = None) -> "LieGroup":
        the_inverse = self._inverse_impl()
        if jacobian is not None:
            LieGroup._check_jacobians_list(jacobian)
            jacobian.append(self._inverse_jacobian(self))
        return the_inverse

    def _compose_jacobian(
        self, group2: "LieGroup"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not type(self) is type(group2):
            raise ValueError("Lie groups for compose must be of the same type.")
        g2_inverse = group2._inverse_impl()
        jac1 = g2_inverse.adjoint()
        jac2 = (
            torch.eye(group2.dof(), dtype=self.dtype)
            .repeat(group2.shape[0], 1, 1)
            .to(group2.device)
        )
        return jac1, jac2

    @staticmethod
    def _inverse_jacobian(group: "LieGroup") -> torch.Tensor:
        return -group.adjoint()

    def _local_impl(self, variable2: Manifold) -> torch.Tensor:
        variable2 = cast(LieGroup, variable2)
        return self.between(variable2).log_map()

    def _local_jacobian(self, variable2: Manifold) -> Tuple[torch.Tensor, torch.Tensor]:
        # Need the log_map derivative which is not yet implemented
        raise NotImplementedError

    def _retract_impl(self, delta: torch.Tensor) -> "LieGroup":
        return self.compose(self.exp_map(delta))

    # added to avoid casting downstream
    def copy(self, new_name: Optional[str] = None) -> "LieGroup":
        return cast(LieGroup, super().copy(new_name=new_name))


# Alias for LieGroup.adjoint()
def adjoint(variable: LieGroup) -> torch.Tensor:
    return variable.adjoint()


def between(
    variable1: LieGroup,
    variable2: LieGroup,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> LieGroup:
    return variable1.between(variable2, jacobians=jacobians)


# Alias for LieGroup.compose()
def compose(
    variable1: LieGroup,
    variable2: LieGroup,
    jacobians: Optional[List[torch.Tensor]] = None,
) -> LieGroup:
    return variable1.compose(variable2, jacobians=jacobians)


# Alias for LieGroup.inverse()
def inverse(
    variable1: LieGroup, jacobian: Optional[List[torch.Tensor]] = None
) -> LieGroup:
    return variable1.inverse(jacobian=jacobian)


# Alias for LieGroup.log_map()
def log_map(variable: LieGroup) -> torch.Tensor:
    return variable.log_map()


# Alias for LieGroup.exp_map()
def exp_map(variable: LieGroup, tangent_vector: torch.Tensor) -> LieGroup:
    return variable.__class__.exp_map(tangent_vector)
