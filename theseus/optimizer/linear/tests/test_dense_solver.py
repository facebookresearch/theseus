# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th


def _create_linear_system(batch_size=32, matrix_size=10):
    A = torch.randn((batch_size, matrix_size, matrix_size))
    AtA = torch.empty((batch_size, matrix_size, matrix_size))
    Atb = torch.empty((batch_size, matrix_size, 1))
    x = torch.randn((batch_size, matrix_size))
    for i in range(batch_size):
        AtA[i] = A[i].t() @ A[i] + 0.1 * torch.eye(matrix_size, matrix_size)
        Atb[i] = AtA[i] @ x[i].unsqueeze(1)
    return AtA, Atb, x


@pytest.mark.parametrize("damp_as_tensor", [True, False])
def test_dense_solvers(damp_as_tensor):
    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    for solver_cls in [th.LUDenseSolver, th.CholeskyDenseSolver]:
        solver = solver_cls(
            void_objective, linearization_kwargs={"ordering": void_ordering}
        )
        solver.linearization.AtA, solver.linearization.Atb, x = _create_linear_system()
        solved_x = solver.solve().squeeze()
        error = torch.norm(x - solved_x, p=float("inf"))
        assert error < 1e-4

        # Test damping
        damping = (
            0.2
            if not damp_as_tensor
            else 0.1 * torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
        )
        damping_eps = 1e-3
        solved_x_with_ellipsoidal_damp = solver.solve(
            damping=damping, ellipsoidal_damping=True, damping_eps=damping_eps
        ).squeeze()
        solved_x_with_spherical_damp = solver.solve(
            damping=damping, ellipsoidal_damping=False
        ).squeeze()
        AtA_ellipsoidal_damp = solver.linearization.AtA.clone()
        AtA_spherical_damp = solver.linearization.AtA.clone()
        batch_size, n, _ = AtA_ellipsoidal_damp.shape
        # Add damping (using loop to make it more visually explicit)
        for i in range(batch_size):
            for j in range(n):
                damp = damping if isinstance(damping, float) else damping[i]
                AtA_ellipsoidal_damp[i, j, j] *= 1 + damp
                AtA_ellipsoidal_damp[i, j, j] += damping_eps
                AtA_spherical_damp[i, j, j] += damp
        # AtA_ellipsoidal_damp += damping_eps
        Atb_check_ellipsoidal = AtA_ellipsoidal_damp.bmm(
            solved_x_with_ellipsoidal_damp.unsqueeze(-1)
        )
        Atb_check_spherical = AtA_spherical_damp.bmm(
            solved_x_with_spherical_damp.unsqueeze(-1)
        )
        error = torch.norm(
            solver.linearization.Atb - Atb_check_ellipsoidal, p=float("inf")
        )
        assert error < 1e-4
        error = torch.norm(
            solver.linearization.Atb - Atb_check_spherical, p=float("inf")
        )
        assert error < 1e-4


def test_handle_singular():
    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    batch_size = 32
    matrix_size = 10
    mid_point = batch_size // 2

    for solver_cls in [th.LUDenseSolver, th.CholeskyDenseSolver]:
        with pytest.warns(RuntimeWarning):
            solver = solver_cls(
                void_objective,
                linearization_kwargs={"ordering": void_ordering},
                check_singular=True,
            )
            (
                solver.linearization.AtA,
                solver.linearization.Atb,
                x,
            ) = _create_linear_system(batch_size=batch_size, matrix_size=10)
            solver.linearization.AtA[:mid_point] = torch.zeros(matrix_size, matrix_size)
            solved_x = solver.solve().squeeze()
            error = (x - solved_x) ** 2
            assert error[mid_point:].mean().item() < 1e-6
            assert solved_x[:mid_point].abs().max() == 0
