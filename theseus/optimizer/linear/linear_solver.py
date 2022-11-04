# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Dict, Optional, Type, Union

import torch

from theseus.core import Objective
from theseus.optimizer import Linearization


class LinearSolver(abc.ABC):
    # linearization_cls is optional, since every linear solver will have a default
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        linearization_kwargs = linearization_kwargs or {}
        self.linearization: Linearization = linearization_cls(
            objective, **linearization_kwargs
        )

    @abc.abstractmethod
    def solve(
        self, damping: Optional[Union[float, torch.Tensor]] = None, **kwargs
    ) -> torch.Tensor:
        pass
