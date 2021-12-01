# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from theseus.optimizer import Optimizer, OptimizerInfo


class TheseusLayer(nn.Module):
    def __init__(
        self,
        optimizer: Optimizer,
    ):
        super().__init__()
        self.objective = optimizer.objective
        self.optimizer = optimizer
        self._objectives_version = optimizer.objective.current_version

    def forward(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        track_best_solution: bool = False,
        verbose: bool = False,
        **optimizer_kwargs
    ) -> Tuple[Dict[str, torch.Tensor], OptimizerInfo]:
        if self._objectives_version != self.objective.current_version:
            raise RuntimeError(
                "The objective was modified after the layer's construction, which is "
                "currently not supported."
            )
        self.objective.update(input_data)
        info = self.optimizer.optimize(
            track_best_solution=track_best_solution, verbose=verbose, **optimizer_kwargs
        )
        values = dict(
            [
                (var_name, var.data)
                for var_name, var in self.objective.optim_vars.items()
            ]
        )
        return values, info

    # Applies to() with given args to all tensors in the objective
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.objective.to(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.objective.device

    @property
    def dtype(self) -> torch.dtype:
        return self.objective.dtype
