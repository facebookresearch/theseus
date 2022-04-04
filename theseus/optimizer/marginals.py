# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch

from theseus.geometry import Manifold


class Marginals:
    def __init__(self, variables: Sequence[Manifold]):
        tot_dof = 0
        for v in variables:
            tot_dof += v.dof()
        self.tot_dof = tot_dof

        self.precision = torch.zeros(
            self.tot_dof, self.tot_dof, dtype=variables[0].dtype
        )

    def dof(self) -> int:
        return self.tot_dof
