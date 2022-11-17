# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from typing import Optional, List
from .utils import get_module


class LieGroupAdjoint(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(cls, tangent_vector: torch.Tensor):
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector, jacobians=None):
        pass


class LieGroupCompose(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(
        cls,
        g0: torch.Tensor,
        g1: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, g0, g1, jacobians=None):
        pass


class LieGroupExpMap(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(
        cls,
        tangent_vector: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ):
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector, jacobians=None):
        pass


class LieGroupHat(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(cls, tangent_vector: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector):
        return cls.call(tangent_vector)


class LieGroupInverse(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(
        cls,
        tangent_vector: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ):
        pass

    @classmethod
    def jacobian(cls, group: torch.Tensor) -> torch.Tensor:
        module = get_module(cls)
        if not module.check_group_tensor(group):
            raise ValueError(f"Invalid data tensor for {module.name()}")
        return -module.adjoint(group)

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector, jacobians=None):
        pass


class LieGroupProject(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(cls, matrix: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, matrix):
        return cls.call(matrix)


class LieGroupVee(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(cls, matrix: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, matrix):
        return cls.call(matrix)
