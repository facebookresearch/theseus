# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from typing import Optional, List


class LieGroupAdjoint(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def call(cls, tangent_vector: torch.Tensor):
        pass

    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector, jacobians=None):
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
