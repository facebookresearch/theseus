# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch
import numpy as np
from torch.autograd import gradcheck
from theseus.optimizer.autograd import BaspachoSolveFunction

from theseus.utils import random_sparse_binary_matrix, split_into_param_sizes

import theseus as th

try:
    import theseus.extlib.baspacho_solver  # noqa: F401

    BASPACHO_EXT_NOT_AVAILABLE = False
except ModuleNotFoundError:
    BASPACHO_EXT_NOT_AVAILABLE = True

requires_baspacho = pytest.mark.skipif(
    BASPACHO_EXT_NOT_AVAILABLE,
    reason="Baspacho solver not in theseus extension library",
)


def check_sparse_backward_step(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill, dev="cpu"
):
    torch.manual_seed(hash(str([batch_size, rows_to_cols_ratio, num_cols, fill])))

    # this is necessary assumption, so that the hessian can be full rank. actually we
    # add some damping to At*A's diagonal, so not really necessary
    assert rows_to_cols_ratio >= 1.0
    num_rows = round(rows_to_cols_ratio * num_cols)

    if isinstance(param_size_range, str):
        param_size_range = [int(x) for x in param_size_range.split(":")]

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.BaspachoSparseSolver(
        void_objective, linearization_kwargs={"ordering": void_ordering}, damping=0.01
    )
    linearization = solver.linearization

    A_skel = random_sparse_binary_matrix(
        num_rows, num_cols, fill, min_entries_per_col=1
    )
    void_objective._batch_size = batch_size
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_col_ind = A_skel.indices
    linearization.A_row_ptr = A_skel.indptr
    linearization.A_val = torch.rand((batch_size, A_skel.nnz), dtype=torch.double).to(
        dev
    )
    linearization.b = torch.randn((batch_size, num_rows), dtype=torch.double).to(dev)

    # also need: var dims and var_start_cols (because baspacho is blockwise)
    linearization.var_dims = split_into_param_sizes(num_cols, *param_size_range)
    linearization.var_start_cols = np.cumsum([0, *linearization.var_dims[:-1]])

    linearization.A_val.requires_grad = True
    linearization.b.requires_grad = True
    # Only need this line for the test since the objective is a mock
    solver.reset(dev=dev)
    damping_alpha_beta = (0.023, 0.157)
    inputs = (
        linearization.A_val,
        linearization.b,
        linearization.structure(),
        solver.A_rowPtr,
        solver.A_colInd,
        solver.symbolic_decomposition,
        damping_alpha_beta,
    )

    assert gradcheck(BaspachoSolveFunction.apply, inputs, eps=1e-5, atol=1e-5)


@requires_baspacho
@pytest.mark.baspacho
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.5])
@pytest.mark.parametrize("num_cols", [15, 20])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13"])
@pytest.mark.parametrize("fill", [0.04, 0.07])
def test_sparse_backward_step_cpu(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_sparse_backward_step(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cpu",
    )


@requires_baspacho
@pytest.mark.cudaext
@pytest.mark.baspacho
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.5])
@pytest.mark.parametrize("num_cols", [15, 20])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13"])
@pytest.mark.parametrize("fill", [0.04, 0.07])
def test_sparse_backward_step_cuda(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_sparse_backward_step(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cuda",
    )
