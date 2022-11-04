# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from scipy.sparse import csr_matrix, lil_matrix
import torch


def random_sparse_binary_matrix(
    rows: int, cols: int, fill: float, min_entries_per_col: int, rng: torch.Generator
) -> csr_matrix:
    retv = lil_matrix((rows, cols))

    if min_entries_per_col > 0:
        min_entries_per_col = min(rows, min_entries_per_col)
        rows_array = torch.arange(rows, device=rng.device)
        rows_array_f = rows_array.to(dtype=torch.float)
        for c in range(cols):
            row_selection = rows_array[
                rows_array_f.multinomial(min_entries_per_col, generator=rng)
            ].cpu()
            for r in row_selection:
                retv[r, c] = 1.0

    # make sure last row is non-empty, so: len(indptr) = rows+1
    retv[rows - 1, int(torch.randint(cols, (), device=rng.device, generator=rng))] = 1.0

    num_entries = int(fill * rows * cols)
    while retv.getnnz() < num_entries:
        col = int(torch.randint(cols, (), device=rng.device, generator=rng))
        row = int(torch.randint(rows, (), device=rng.device, generator=rng))
        retv[row, col] = 1.0

    return retv.tocsr()


def split_into_param_sizes(
    n: int, param_size_range_min: int, param_size_range_max: int, rng: torch.Generator
) -> List[int]:
    paramSizes = []
    tot = 0
    while tot < n:
        newParam = min(
            int(
                torch.randint(
                    param_size_range_min,
                    param_size_range_max,
                    (),
                    device=rng.device,
                    generator=rng,
                )
            ),
            n - tot,
        )
        tot += newParam
        paramSizes.append(newParam)
    return paramSizes
