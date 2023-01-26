# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Protocol, Sequence

import torch

import theseus.labs.lie_functional.se3 as _se3_impl
import theseus.labs.lie_functional.so3 as _so3_impl
from .constants import DeviceType
from .lie_group import BinaryOperatorFactory, UnaryOperatorFactory

_CheckFnType = Callable[[torch.Tensor], None]


class _RandFnType(Protocol):
    def __call__(
        *size: int,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
        device: DeviceType = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass


# Namespace to facilitate type-checking downstream
class LieGroupFns:
    def __init__(self, module):
        self.exp, self.jexp = UnaryOperatorFactory(module, "exp")
        self.log, self.jlog = UnaryOperatorFactory(module, "log")
        self.adj = UnaryOperatorFactory(module, "adjoint")
        self.inv, self.jinv = UnaryOperatorFactory(module, "inverse")
        self.hat = UnaryOperatorFactory(module, "hat")
        self.vee = UnaryOperatorFactory(module, "vee")
        self.compose, self.jcompose = BinaryOperatorFactory(module, "compose")
        self.lift = UnaryOperatorFactory(module, "lift")
        self.project = UnaryOperatorFactory(module, "project")
        self.left_act = BinaryOperatorFactory(module, "left_act")
        self.left_project = BinaryOperatorFactory(module, "left_project")
        self.check_group_tensor: _CheckFnType = module.check_group_tensor
        self.check_tangent_vector: _CheckFnType = module.check_tangent_vector
        self.check_hat_matrix: _CheckFnType = module.check_hat_matrix
        if hasattr(module, "check_unit_quaternion"):
            self.check_unit_quaternion: _CheckFnType = module.check_unit_quaternion
        if hasattr(module, "check_lift_matrix"):
            self.check_lift_matrix: _CheckFnType = module.check_lift_matrix
        if hasattr(module, "check_project_matrix"):
            self.check_project_matrix: _CheckFnType = module.check_project_matrix
        self.check_left_act_matrix: _CheckFnType = module.check_left_act_matrix
        self.check_left_project_matrix: _CheckFnType = module.check_left_project_matrix
        self.rand: _RandFnType = module.rand
        self.randn: _RandFnType = module.randn


se3_fns = LieGroupFns(_se3_impl)
so3_fns = LieGroupFns(_so3_impl)
