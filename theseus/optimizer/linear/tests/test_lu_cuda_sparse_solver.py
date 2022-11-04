# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th


def _build_sparse_mat(batch_size, rng):
    all_cols = list(range(10))
    col_ind = []
    row_ptr = [0]
    for i in range(12):
        start = max(0, i - 2)
        end = min(i + 1, 10)
        col_ind += all_cols[start:end]
        row_ptr.append(len(col_ind))
    data = torch.randn((batch_size, len(col_ind)), dtype=torch.double, generator=rng)
    return 12, 10, data, col_ind, row_ptr


@pytest.mark.cudaext
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("float_damping", [True, False])
def test_sparse_solver(batch_size: int, float_damping: bool):
    rng = torch.Generator()
    rng.manual_seed(0)
    if not torch.cuda.is_available():
        return

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.LUCudaSparseSolver(
        void_objective,
        linearization_kwargs={"ordering": void_ordering},
    )
    linearization = solver.linearization

    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size, rng)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_val = data.cuda()
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr
    linearization.b = torch.randn(
        (batch_size, num_rows), dtype=torch.double, generator=rng
    ).cuda()
    # Only need this line for the test since the objective is a mock
    solver.reset(batch_size=batch_size)

    if float_damping:
        damping = 1e-4
    else:
        damping = 0.01 * torch.rand(batch_size, generator=rng)  # type: ignore
    solved_x = solver.solve(damping=damping, ellipsoidal_damping=False)

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :].cpu())
        Ai = torch.tensor(csrAi.todense(), dtype=torch.double)
        ata = Ai.T @ Ai
        b = linearization.b[i].cpu()
        atb = torch.DoubleTensor(csrAi.transpose() @ b)

        # the linear system solved is with matrix AtA
        solved_xi_cpu = solved_x[i].cpu()
        damp = damping if float_damping else damping[i]  # type: ignore
        atb_check = ata @ solved_xi_cpu + damp * solved_xi_cpu

        torch.testing.assert_close(atb, atb_check, atol=1e-2, rtol=1e-2)


def check_sparse_solver_multistep(batch_size: int, test_exception: bool):
    rng = torch.Generator()
    rng.manual_seed(37)

    if not torch.cuda.is_available():
        return

    num_steps = 3
    torch.manual_seed(37)

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.LUCudaSparseSolver(
        void_objective,
        linearization_kwargs={"ordering": void_ordering},
        num_solver_contexts=(num_steps - 1) if test_exception else num_steps,
    )
    linearization = solver.linearization

    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size, rng)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr

    # Only need this line for the test since the objective is a mock
    solver.reset(batch_size=batch_size)

    As = [
        torch.randn(
            (batch_size, len(col_ind)), dtype=torch.double, generator=rng
        ).cuda()
        for _ in range(num_steps)
    ]
    bs = [
        torch.randn((batch_size, num_rows), dtype=torch.double, generator=rng).cuda()
        for _ in range(num_steps)
    ]
    c = torch.randn((batch_size, num_cols), dtype=torch.double, generator=rng).cuda()

    # batched dot product
    def batched_dot(a, b):
        return torch.sum(a * b, dim=1)

    # computes accum = sum(A_i \ b_i), returns dot(accum, c)
    def iterate_solver(As, bs):
        accum = None
        for A, b in zip(As, bs):
            linearization.A_val = A
            linearization.b = b
            res = solver.solve()
            accum = res if accum is None else (accum + res)
        return batched_dot(c, accum)

    for A, b in zip(As, bs):
        A.requires_grad = True
        b.requires_grad = True

    result = iterate_solver(As, bs)

    # if insufficient contexts, assert exception is raised
    if test_exception:
        with pytest.raises(RuntimeError):
            result.backward(torch.ones_like(result))
        return

    # otherwise, compute and check gradient
    result.backward(torch.ones_like(result))

    # we select random vectors `perturb` and check if the (numerically
    # approximated) directional derivative matches with dot(perturb, grad)
    epsilon = 1e-7
    num_checks = 10
    for i in range(num_checks):
        for perturb_A in [False, True]:
            for step in range(num_steps):
                perturbed_As = [A.detach().clone() for A in As]
                perturbed_bs = [b.detach().clone() for b in bs]

                if perturb_A:
                    perturb = torch.randn(
                        (batch_size, len(col_ind)), dtype=torch.double, generator=rng
                    ).cuda()
                    perturbed_As[step] += perturb * epsilon
                    analytic_der = batched_dot(perturb, As[step].grad)
                else:
                    perturb = torch.randn(
                        (batch_size, num_rows), dtype=torch.double, generator=rng
                    ).cuda()
                    perturbed_bs[step] += perturb * epsilon
                    analytic_der = batched_dot(perturb, bs[step].grad)

                perturbed_result = iterate_solver(perturbed_As, perturbed_bs)
                numeric_der = (perturbed_result - result) / epsilon
                torch.testing.assert_close(
                    numeric_der, analytic_der, rtol=1e-3, atol=1e-3
                )


@pytest.mark.cudaext
@pytest.mark.parametrize("batch_size", [1, 32])
def test_sparse_solver_multistep_gradient(batch_size):
    check_sparse_solver_multistep(batch_size, False)


@pytest.mark.cudaext
@pytest.mark.parametrize("batch_size", [1, 32])
def test_sparse_solver_multistep_exception(batch_size):
    check_sparse_solver_multistep(batch_size, True)
