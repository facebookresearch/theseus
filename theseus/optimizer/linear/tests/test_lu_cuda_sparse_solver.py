import pytest  # noqa: F401
import torch

import theseus as th


def _build_sparse_mat(batch_size):
    all_cols = list(range(10))
    col_ind = []
    row_ptr = [0]
    for i in range(12):
        start = max(0, i - 2)
        end = min(i + 1, 10)
        col_ind += all_cols[start:end]
        row_ptr.append(len(col_ind))
    data = torch.randn((batch_size, len(col_ind)), dtype=torch.double)
    return 12, 10, data, col_ind, row_ptr


@pytest.mark.cudaext
def test_sparse_solver():

    if not torch.cuda.is_available():
        return

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.LUCudaSparseSolver(
        void_objective,
        linearization_kwargs={"ordering": void_ordering},
    )
    linearization = solver.linearization

    batch_size = 4
    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_val = data.cuda()
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr
    linearization.b = torch.randn((batch_size, num_rows), dtype=torch.double).cuda()
    # Only need this line for the test since the objective is a mock
    solver.reset(batch_size=batch_size)

    solved_x = solver.solve()

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :].cpu())
        Ai = torch.tensor(csrAi.todense(), dtype=torch.double)
        ata = Ai.T @ Ai
        b = linearization.b[i].cpu()
        atb = torch.Tensor(csrAi.transpose() @ b)

        # the linear system solved is with matrix AtA
        atb_check = ata @ solved_x[i].cpu()

        max_offset = torch.norm(atb - atb_check, p=float("inf"))
        assert max_offset < 1e-4


def check_sparse_solver_multistep(test_exception: bool):

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

    batch_size = 4
    void_objective._batch_size = batch_size
    num_rows, num_cols, data, col_ind, row_ptr = _build_sparse_mat(batch_size)
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_col_ind = col_ind
    linearization.A_row_ptr = row_ptr

    # Only need this line for the test since the objective is a mock
    solver.reset(batch_size=batch_size)

    As = [
        torch.randn((batch_size, len(col_ind)), dtype=torch.double).cuda()
        for _ in range(num_steps)
    ]
    bs = [
        torch.randn((batch_size, num_rows), dtype=torch.double).cuda()
        for _ in range(num_steps)
    ]
    c = torch.randn((batch_size, num_cols), dtype=torch.double).cuda()

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
                        (batch_size, len(col_ind)), dtype=torch.double
                    ).cuda()
                    perturbed_As[step] += perturb * epsilon
                    analytic_der = batched_dot(perturb, As[step].grad)
                else:
                    perturb = torch.randn(
                        (batch_size, num_rows), dtype=torch.double
                    ).cuda()
                    perturbed_bs[step] += perturb * epsilon
                    analytic_der = batched_dot(perturb, bs[step].grad)

                perturbed_result = iterate_solver(perturbed_As, perturbed_bs)
                numeric_der = (perturbed_result - result) / epsilon
                assert numeric_der.isclose(analytic_der, rtol=1e-4, atol=1e-4).all()


@pytest.mark.cudaext
def test_sparse_solver_multistep_gradient():
    check_sparse_solver_multistep(False)


@pytest.mark.cudaext
def test_sparse_solver_multistep_exception():
    check_sparse_solver_multistep(True)
