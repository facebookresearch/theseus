# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from theseus.core import Objective, Vectorize


# All info information is batched
@dataclass
class OptimizerInfo:
    best_solution: Optional[Dict[str, torch.Tensor]]
    # status for each element
    status: np.ndarray


class Optimizer(abc.ABC):
    def __init__(self, objective: Objective, *args, vectorize: bool = True, **kwargs):
        self.objective = objective
        if vectorize:
            Vectorize(
                self.objective, empty_cuda_cache=kwargs.get("empty_cuda_cache", False)
            )
        self._objectives_version = objective.current_version

    @abc.abstractmethod
    def _optimize_impl(self, **kwargs) -> OptimizerInfo:
        pass

    def optimize(self, **kwargs) -> OptimizerInfo:
        if self._objectives_version != self.objective.current_version:
            raise RuntimeError(
                "The objective was modified after optimizer construction, which is "
                "currently not supported."
            )
        return self._optimize_impl(**kwargs)
