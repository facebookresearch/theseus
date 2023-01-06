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

Test cases:
- with / without vectorization
- with / without factor to variable message damping
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
    vectorize,
    optimize_kwargs,
):
    max_iterations = 20

    n_variables = 100
    batch_size = 1

    rng = torch.Generator()
    rng.manual_seed(0)

    variables = []
    meas_vars = []
    for i in range(n_variables):
        variables.append(
            th.Vector(tensor=torch.rand(batch_size, 1, generator=rng), name=f"x_{i}")
        )
        meas_vars.append(
            th.Vector(
                tensor=torch.rand(batch_size, 1, generator=rng), name=f"meas_x{i}"
            )
        )

    objective = th.Objective()
    # measurement cost functions
    meas_weight = th.ScaleCostWeight(1.0, name="meas_weight")
    for var, meas in zip(variables, meas_vars):
        objective.add(th.Difference(var, meas, meas_weight))
    # smoothness cost functions between adjacent variables and between random variables
    smoothness_weight = th.ScaleCostWeight(4.0, name="smoothness_weight")
    zero = th.Vector(tensor=torch.zeros(batch_size, 1), name="zero")
    for i in range(n_variables - 1):
        objective.add(
            th.Between(variables[i], variables[i + 1], zero, smoothness_weight)
        )
    for i in range(100):
        ix1, ix2 = torch.randint(n_variables, (2,))
        # ix1, ix2 = 0, 2
        objective.add(
            th.Between(variables[ix1], variables[ix2], zero, smoothness_weight)
        )

    # initial input tensors
    measurements = torch.rand(batch_size, n_variables, generator=rng)
    input_tensors = {}
    for var in variables:
        input_tensors[var.name] = var.tensor
    for i in range(len(measurements[0])):
        input_tensors[f"meas_x{i}"] = measurements[:, i][:, None]

    # GBP inference
    # optimizer = th.GaussianBeliefPropagation(
    #     objective, max_iterations=max_iterations, vectorize=vectorize
    # )
    # optimizer.set_params(max_iterations=max_iterations)
    # objective.update(input_tensors)
    # initial_error = objective.error_squared_norm() / 2

    # callback_expected_iter = [0]

    # def callback(opt_, info_, None, it_):
    #     assert opt_ is optimizer
    #     assert isinstance(info_, th.optimizer.OptimizerInfo)
    #     assert it_ == callback_expected_iter[0]
    #     callback_expected_iter[0] += 1

    # info = optimizer.optimize(
    #     track_best_solution=True,
    #     track_err_history=True,
    #     end_iter_callback=callback,
    #     verbose=True,
    #     **optimize_kwargs,
    # )
    # gbp_solution = [var.tensor.clone() for var in variables]

    # Solve with Gauss-Newton

    def gn_callback(opt_, info_, _, it_):
        out = list(opt_.objective.optim_vars.values())
        vec = torch.cat([v.tensor for v in out])
        print(vec.flatten())

    objective.update(input_tensors)
    gn_optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver)
    gn_optimizer.set_params(max_iterations=max_iterations)
    gn_optimizer.optimize(verbose=True, end_iter_callback=gn_callback)
    gn_solution = [var.tensor.clone() for var in variables]

    # Solve with linear solver
    objective.update(input_tensors)
    linear_optimizer = th.LinearOptimizer(objective, th.CholeskyDenseSolver)
    linear_optimizer.optimize(verbose=True)
    lin_solution = [var.tensor.clone() for var in variables]
    lin_solve_err = objective.error_squared_norm() / 2
    print("linear solver error", lin_solve_err.item())

    for x, x_target in zip(lin_solution, gn_solution):
        print(x, x_target)
        assert x.allclose(x_target)

    # print("comparing GBP")
    # for x, x_target in zip(gbp_solution, gn_solution):
    #     print(x, x_target)
    #     assert x.allclose(x_target)

    # for x, x_target in zip(gbp_solution, lin_solution):
    #     print(x, x_target)
    #     assert x.allclose(x_target)

    # _check_info(info, batch_size, max_iterations, initial_error, objective)

    # # Visualise reconstructed surface
    # soln_vec = torch.cat(gbp_solution, dim=1)[0]
    # import matplotlib.pylab as plt
    # plt.scatter(torch.arange(n_variables), soln_vec, label="solution")
    # plt.scatter(torch.arange(n_variables), measurements[0], label="meas")
    # plt.legend()
    # plt.show()


def test_gbp_linear_solver():
    optimize_kwargs = {}

    # run_gbp_linear_solver(vectorize=True, optimize_kwargs=optimize_kwargs)
    run_gbp_linear_solver(vectorize=False, optimize_kwargs=optimize_kwargs)
