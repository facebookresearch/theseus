# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from typing import List, Optional
from .utils import check_jacobians_list


class ExpMap(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector):
        pass


def UnaryFunctionFactory(module, func_name):
    fn_base = getattr(module, "_" + func_name + "_base")
    j_fn_base = getattr(module, "_j_" + func_name + "_base")

    def func(
        input: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            check_jacobians_list(jacobians)
            jacobians.append(j_fn_base(input))
        return fn_base(input)

    return func
