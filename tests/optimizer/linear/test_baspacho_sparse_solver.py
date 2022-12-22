# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

from tests.extlib.common import run_if_baspacho
from tests.optimizer.autograd.test_baspacho_sparse_backward import (
    get_linearization_and_solver_for_random_sparse,
)


def check_sparse_solver(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill, dev="cpu"
):
    rng = torch.Generator(device=dev)
    rng.manual_seed(hash(str([batch_size, rows_to_cols_ratio, num_cols, fill])))

    linearization, solver = get_linearization_and_solver_for_random_sparse(
        batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill, dev, rng
    )

    # Only need this line for the test since the objective is a mock
    solver.reset(dev=dev)
    damping = 1e-4
    solved_x = solver.solve(damping=damping, ellipsoidal_damping=False)

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :].cpu())
        Ai = torch.tensor(csrAi.todense(), dtype=torch.double)
        ata = Ai.T @ Ai
        b = linearization.b[i].cpu()
        atb = torch.DoubleTensor(csrAi.transpose() @ b)

        # the linear system solved is with matrix AtA
        solved_xi_cpu = solved_x[i].cpu()
        atb_check = ata @ solved_xi_cpu + damping * solved_xi_cpu
        torch.testing.assert_close(atb, atb_check, atol=1e-3, rtol=1e-3)


@run_if_baspacho()
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.1, 1.7])
@pytest.mark.parametrize("num_cols", [30, 70])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13"])
@pytest.mark.parametrize("fill", [0.02, 0.05])
def test_baspacho_solver_cpu(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_sparse_solver(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cpu",
    )


@run_if_baspacho()
@pytest.mark.cudaext
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.1, 1.7])
@pytest.mark.parametrize("num_cols", [30, 70])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13"])
@pytest.mark.parametrize("fill", [0.02, 0.05])
def test_baspacho_solver_cuda(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_sparse_solver(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cuda",
    )
