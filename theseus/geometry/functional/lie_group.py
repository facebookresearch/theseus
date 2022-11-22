# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from typing import List, Tuple, Optional
from .utils import check_jacobians_list

# There are four functions associated with each Lie group operator xxx.
# _xxx_impl: analytic implementation of the operator, return xxx
# _jxxx_impl: analytic implementation of the operator jacobian, return jxxx and xxx
# _xxx_autograd_fn: a torch.autograd.Function wrapper of _xxx_impl
# _jxxx_autograd_fn: simply equivalent to _jxxx_impl for now


class UnaryOperator(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, input):
        pass


def UnaryOperatorFactory(module, op_name):
    op_autograd_fn = getattr(module, "_" + op_name + "_autograd_fn")
    jop_autograd_fn = getattr(module, "_j" + op_name + "_autograd_fn")

    def op(
        input: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            check_jacobians_list(jacobians)
            jacobians.append(jop_autograd_fn(input)[0])
        return op_autograd_fn(input)

    def jop(input: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return jop_autograd_fn(input)

    return op, jop
