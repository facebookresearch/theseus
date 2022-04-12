# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import count
from typing import Optional

from .cost_function import CostFunction
from .variable import Variable


class Batch:
    _ids = count(0)

    def __init__(self) -> None:
        self._id = next(Batch._ids)
        self._name: str = ""

    def __str__(self):
        return self._name

    @property
    def name(self) -> str:
        return self._name


class CostFunctionBatch(Batch):
    def __init__(self, cost_function: CostFunction, name: Optional[str] = None) -> None:
        super().__init__()
        if name is None:
            self._name = f"{cost_function.__class__.__name__}__batch__{self._id}"
        else:
            self._name = name


class VariableBatch(Batch):
    def __init__(self, variable: Variable) -> None:
        super().__init__()
        self._name = f"{variable.__class__.__name__}__batch__{self._id}"
