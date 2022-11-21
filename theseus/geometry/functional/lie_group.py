# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from typing import List, Optional
from .utils import check_jacobians_list

# There are four functions associated with each Lie group operator xxx.
# _xxx_impl: mathematical implementation of the operator
# _j_xxx_impl: mathematical implementation of the operator jacobian
# _xxx_base: a torch.autograd.Function wrapper of _xxx_impl
# _j_xxx_base: simply equivalent to _j_xxx_impl


class UnaryOperator(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, input):
        pass


def UnaryOperatorFactory(module, op_name):
    op_base = getattr(module, "_" + op_name + "_base")
    j_op_base = getattr(module, "_j_" + op_name + "_base")

    def op(
        input: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            check_jacobians_list(jacobians)
            jacobians.append(j_op_base(input))
        return op_base(input)

    return op
