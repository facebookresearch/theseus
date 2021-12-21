# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def random_sparse_binary_matrix(rows, cols, fill, min_entries_per_col) -> csr_matrix:
    retv = lil_matrix((rows, cols))

    if min_entries_per_col > 0:
        min_entries_per_col = min(rows, min_entries_per_col)
        rows_array = np.arange(rows)
        for c in range(cols):
            for r in np.random.choice(rows_array, min_entries_per_col):
                retv[r, c] = 1.0

    num_entries = int(fill * rows * cols)
    while retv.getnnz() < num_entries:
        col = np.random.randint(cols)
        row = np.random.randint(rows)
        retv[row, col] = 1.0

    return retv.tocsr()
