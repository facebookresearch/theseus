# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from theseus.constants import __FROM_THESEUS_LAYER_TOKEN__
from theseus.core import Objective, Vectorize


# All info information is batched
@dataclass
class OptimizerInfo:
    best_solution: Optional[Dict[str, torch.Tensor]]
    # status for each element
    status: np.ndarray


class Optimizer(abc.ABC):
    def __init__(self, objective: Objective, *args, vectorize: bool = False, **kwargs):
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
        from_theseus_layer = kwargs.get(__FROM_THESEUS_LAYER_TOKEN__, False)
        if not from_theseus_layer and not self.objective.vectorized:
            warnings.warn(
                "Vectorization is off by default when not running from TheseusLayer. "
                "Using TheseusLayer is the recommended way to run our optimizers."
            )
        if self._objectives_version != self.objective.current_version:
            raise RuntimeError(
                "The objective was modified after optimizer construction, which is "
                "currently not supported."
            )
        return self._optimize_impl(**kwargs)
