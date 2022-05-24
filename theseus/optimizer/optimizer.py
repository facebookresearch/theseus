import abc
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from theseus.core import Objective


# All info information is batched
@dataclass
class OptimizerInfo:
    best_solution: Optional[Dict[str, torch.Tensor]]
    # status for each element
    status: np.ndarray


class Optimizer(abc.ABC):
    def __init__(self, objective: Objective, *args, **kwargs):
        self.objective = objective
        if not self.objective.is_setup:
            self.objective.setup()
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
