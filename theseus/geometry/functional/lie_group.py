# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import abc

from typing import List, Optional, Tuple
from .utils import check_jacobians_list


class ExpMap(torch.autograd.Function):
    @classmethod
    @abc.abstractmethod
    def forward(cls, ctx, tangent_vector):
        pass


def ExpMapFactory(module):
    def exp_map(
        tangent_vector: torch.Tensor,
        jacobians: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if jacobians is not None:
            check_jacobians_list(jacobians)
            jacobians.append(module._j_exp_map_impl(tangent_vector))
        return module.ExpMap.apply(tangent_vector)

    def jexp_map(tangent_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return module.ExpMap.apply(tangent_vector), module._j_exp_map_impl(
            tangent_vector
        )

    return exp_map, jexp_map
