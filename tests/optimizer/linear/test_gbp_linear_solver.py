# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import theseus as th


"""
Build linear 1D surface estimation problem.
Solve using GBP and using matrix inversion and compare answers.
GBP exactly computes the marginal means on convergence.

All the following cases should not affect the converged solution:
- with / without vectorization
- with / without factor to variable message damping
- with / without dropout
- with / without factor linear system damping
"""


def _check_info(info, batch_size, max_iterations, initial_error, objective):
    assert info.err_history.shape == (batch_size, max_iterations + 1)
    assert info.err_history[:, 0].allclose(initial_error)
    assert info.err_history.argmin(dim=1).allclose(info.best_iter + 1)
    last_error = objective.error_squared_norm() / 2
    last_convergence_idx = info.converged_iter.max().item()
    assert info.err_history[:, last_convergence_idx].allclose(last_error)


def run_gbp_linear_solver(
    frac_loops,
    vectorize=True,
    ftov_damping=0.0,
    dropout=0.0,
    lin_system_damping=torch.tensor([1e-4]),
):
    max_iterations = 200

    n_variables = 100
    batch_size = 1

    torch.manual_seed(0)

    # initial input tensors
    # measurements come from x = sin(t / 50) * t**2 / 250 + 1 with random noise added
    ts = torch.arange(n_variables)
    true_meas = torch.sin(ts / 10.0) * ts * ts / 250.0 + 1
    noisy_meas = true_meas[None, :].repeat(batch_size, 1)
    noisy_meas += torch.normal(torch.zeros_like(noisy_meas), 1.0)

    variables = []
    meas_vars = []
    for i in range(n_variables):
        variables.append(th.Vector(tensor=torch.rand(batch_size, 1), name=f"x_{i}"))
        meas_vars.append(th.Vector(tensor=torch.rand(batch_size, 1), name=f"meas_x{i}"))

    objective = th.Objective()

    # measurement cost functions
    meas_weight = th.ScaleCostWeight(5.0, name="meas_weight")
    for var, meas in zip(variables, meas_vars):
        objective.add(th.Difference(var, meas, meas_weight))

    # smoothness cost functions between adjacent variables
    smoothness_weight = th.ScaleCostWeight(2.0, name="smoothness_weight")
    zero = th.Vector(tensor=torch.zeros(batch_size, 1), name="zero")
    for i in range(n_variables - 1):
        objective.add(
            th.Between(variables[i], variables[i + 1], zero, smoothness_weight)
        )

    # difference cost functions between non-adjacent variables to give
    # off diagonal elements in information matrix
    difference_weight = th.ScaleCostWeight(1.0, name="difference_weight")
    for i in range(int(n_variables * frac_loops)):
        ix1, ix2 = torch.randint(n_variables, (2,))
        diff = th.Vector(
            tensor=torch.tensor([[true_meas[ix2] - true_meas[ix1]]]), name=f"diff{i}"
        )
        diff.tensor += torch.normal(torch.zeros(1, 1), 0.2)
        objective.add(
            th.Between(variables[ix1], variables[ix2], diff, difference_weight)
        )

    input_tensors = {}
    for var in variables:
        input_tensors[var.name] = var.tensor
    for i in range(len(noisy_meas[0])):
        input_tensors[f"meas_x{i}"] = noisy_meas[:, i][:, None]

    # Solve with GBP
    optimizer = th.GaussianBeliefPropagation(
        objective, max_iterations=max_iterations, vectorize=vectorize
    )
    optimizer.set_params(max_iterations=max_iterations)
    objective.update(input_tensors)
    initial_error = objective.error_squared_norm() / 2

    callback_expected_iter = [0]

    def callback(opt_, info_, _, it_):
        assert opt_ is optimizer
        assert isinstance(info_, th.optimizer.OptimizerInfo)
        assert it_ == callback_expected_iter[0]
        callback_expected_iter[0] += 1

    info = optimizer.optimize(
        track_best_solution=True,
        track_err_history=True,
        end_iter_callback=callback,
        ftov_msg_damping=ftov_damping,
        dropout=dropout,
        lin_system_damping=lin_system_damping,
        verbose=True,
    )
    gbp_solution = [var.tensor.clone() for var in variables]

    # Solve with linear solver
    objective.update(input_tensors)
    linear_optimizer = th.LinearOptimizer(objective, th.CholeskyDenseSolver)
    linear_optimizer.optimize(verbose=True)
    lin_solution = [var.tensor.clone() for var in variables]

    # Solve with Gauss-Newton
    # If problem is poorly conditioned solving with Gauss-Newton can yield
    # a slightly different solution to one linear solve, so check both
    objective.update(input_tensors)
    gn_optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver)
    gn_optimizer.optimize(verbose=True)
    gn_solution = [var.tensor.clone() for var in variables]

    # checks
    for x, x_target in zip(gbp_solution, lin_solution):
        assert x.allclose(x_target, rtol=1e-3)
    for x, x_target in zip(gbp_solution, gn_solution):
        assert x.allclose(x_target, rtol=1e-3)
    _check_info(info, batch_size, max_iterations, initial_error, objective)

    # # Visualise reconstructed surface
    # soln_vec = torch.cat(gbp_solution, dim=1)[0]
    # import matplotlib.pylab as plt
    # plt.scatter(torch.arange(n_variables), soln_vec, label="solution")
    # plt.scatter(torch.arange(n_variables), noisy_meas[0], label="meas")
    # plt.legend()
    # plt.show()


def test_gbp_linear_solver():

    # problems with increasing loopyness
    # the loopier the fewer iterations to solve
    frac_loops = [0.1, 0.2, 0.5]
    for frac in frac_loops:

        run_gbp_linear_solver(frac_loops=frac)

        # with factor to variable message damping, may take too many steps to converge
        # run_gbp_linear_solver(vectorize=vectorize, frac_loops=frac, ftov_damping=0.1)
        # with dropout
        run_gbp_linear_solver(frac_loops=frac, dropout=0.1)

        # test linear system damping
        run_gbp_linear_solver(frac_loops=frac, lin_system_damping=torch.tensor([0.0]))
        run_gbp_linear_solver(frac_loops=frac, lin_system_damping=torch.tensor([1e-2]))
        run_gbp_linear_solver(frac_loops=frac, lin_system_damping=torch.tensor([1e-6]))

    # test without vectorization once
    run_gbp_linear_solver(frac_loops=0.5, vectorize=False)
