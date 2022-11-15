# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import abc

import torch


class LieGroupFunction:
    @staticmethod
    @abc.abstractmethod
    def dim() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def check_group_tensor(tensor: torch.Tensor) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def check_tangent_vector(tangent_vector: torch.Tensor) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def check_hat_matrix(matrix: torch.Tensor):
        pass

    @staticmethod
    @abc.abstractmethod
    def rand(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass

    @staticmethod
    @abc.abstractmethod
    def randn(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass

    class project(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(cls, matrix: torch.Tensor) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, matrix):
            return cls.call(matrix)

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class left_project(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @classmethod
        def call(cls, group: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
            manifold = cls.manifold()
            return manifold.project.call(
                manifold.left_apply.call(manifold.inverse.call(group), matrix)
            )

        @classmethod
        def forward(cls, ctx, group, matrix):
            return cls.call(group, matrix)

    class right_project(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @classmethod
        def call(cls, matrix: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
            manifold = cls.manifold()
            return manifold.project.call(
                manifold.right_apply.call(matrix, manifold.inverse.call(group))
            )

        @staticmethod
        def forward(cls, ctx, group, matrix):
            return cls.call(group, matrix)

    class left_apply(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(
            cls,
            group: torch.Tensor,
            matrix: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, group, matrix, jacobians):
            pass

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class right_apply(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(
            cls,
            matrix: torch.Tensor,
            group: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, matrix, group, jacobians):
            pass

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class hat(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(cls, tangent_vector: torch.Tensor) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, tangent_vector):
            return LieGroupFunction.call(tangent_vector)

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class vee(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(cls, matrix: torch.Tensor) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, matrix):
            return LieGroupFunction.vee.call(matrix)

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class exp_map(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(
            cls,
            tangent_vector: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def jacobian(cls, tangent_vector: torch.Tensor) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, tangent_vector, jacobians=None):
            pass

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class adjoint(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(cls, g: torch.Tensor) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, g):
            pass

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class inverse(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(
            cls, g: torch.Tensor, jacobians: Optional[List[torch.Tensor]] = None
        ) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def jacobian(cls, g: torch.Tensor) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, g, jacobians=None):
            pass

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass

    class compose(torch.autograd.Function):
        @staticmethod
        @abc.abstractmethod
        def manifold():
            pass

        @abc.abstractclassmethod
        def call(
            cls,
            g0: torch.Tensor,
            g1: torch.Tensor,
            jacobians: Optional[List[torch.Tensor]] = None,
        ) -> torch.Tensor:
            pass

        @abc.abstractclassmethod
        def forward(cls, ctx, g0, g1, jacobians=None):
            pass

        @abc.abstractclassmethod
        def backward(cls, ctx, grad_output):
            pass
