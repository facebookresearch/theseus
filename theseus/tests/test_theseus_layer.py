# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import mock
import pytest  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F

import theseus as th
import theseus.utils as thutils
from theseus.core.tests.common import (
    MockCostFunction,
    MockCostWeight,
    MockVar,
    create_objective_with_mock_cost_functions,
)
from theseus.theseus_layer import TheseusLayer

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def model(x, b):
    return (b[..., :1] * x) ** 2 + b[..., 1:]


def model_grad(x, b):
    g1 = 2 * (b[..., :1] * x) * x
    g2 = b[..., 1:] * torch.ones_like(x)
    return g1, g2


# This is a cost function of two variables that tries to fit
# f(x;b) = A * x[0]^ 2 + B to the dataset
# given by xs and ys. Here the variables for the cost function
# are A, B, and the goal is to minimize MSE over the
# dataset. We will pass the variables as a single
# variable object of dimension 2.
class QuadraticFitCostFunction(th.CostFunction):
    def __init__(self, optim_vars, cost_weight, xs=None, ys=None):
        super().__init__(cost_weight, name="qf_cost_function")
        assert len(optim_vars) == 1 and optim_vars[0].dof() == 2
        for i, var in enumerate(optim_vars):
            setattr(self, f"optim_var_{i}", var)
            self.register_optim_var(f"optim_var_{i}")
        self.xs = xs
        self.ys = ys

        self._optim_vars = optim_vars

    def error_from_tensors(self, optim_var_0_data):
        pred_y = model(self.xs, optim_var_0_data)
        return self.ys - pred_y

    # err = y - f(x:b), where b are the current variable values
    def error(self):
        return self.error_from_tensors(self.optim_var_0.data)

    def jacobians(self):
        g1, g2 = model_grad(self.xs, self.optim_var_0.data)
        return [-torch.stack([g1, g2], axis=2)], self.error()

    def dim(self):
        return self.xs.shape[1]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.xs = self.xs.to(*args, **kwargs)
        self.ys = self.ys.to(*args, **kwargs)

    def _copy_impl(self, new_name=None):
        return QuadraticFitCostFunction(
            [v.copy() for v in self._optim_vars], self.weight.copy(), self.xs, self.ys
        )


def create_qf_theseus_layer(
    xs,
    ys,
    cost_weight=th.ScaleCostWeight(1.0),
    nonlinear_optimizer_cls=th.GaussNewton,
    linear_solver_cls=th.CholeskyDenseSolver,
    max_iterations=10,
    use_learnable_error=False,
):
    variables = [th.Vector(2, name="coefficients")]
    objective = th.Objective()
    cost_function = QuadraticFitCostFunction(variables, cost_weight, xs=xs, ys=ys)

    if use_learnable_error:
        # For learnable error we embed the original cost weight as an auxiliary
        # variable that's part of the error function, and now becomes a learnable
        # parameter of the error
        def error_fn(optim_vars, aux_vars):
            # aux_vars is the learned weight
            # note that this is a hybrid cost function since, part of the function
            # follows the structure of QuadraticFitCostFunction, only the error weight
            # factor (aux_vars[0]) is learned
            return aux_vars[0].data * cost_function.error_from_tensors(
                optim_vars[0].data
            )

        if isinstance(cost_weight, th.ScaleCostWeight):
            # this case only hits with the reference layer, for which weight
            # is not learned (just a scalar value of 1)
            cost_weight_dim = None  # Vector infers dimension from given cw_data
            cw_data = torch.ones(1, 1)
        elif isinstance(cost_weight, th.DiagonalCostWeight):
            # cw_data is None, since no need to pass data to aux variable,
            # because it will be replaced during forward pass of learned layer
            cost_weight_dim = cost_function.weight.diagonal.shape[1]
            cw_data = None

        # in this case the cost weight is a scalar constant of 1.0
        learnable_cost_function = th.AutoDiffCostFunction(
            variables,
            error_fn,
            cost_function.dim(),
            aux_vars=[
                th.Vector(cost_weight_dim, name="learnable_err_param", data=cw_data)
            ],
            autograd_vectorize=True,
        )
        objective.add(learnable_cost_function)
    else:
        objective.add(cost_function)

    optimizer = nonlinear_optimizer_cls(
        objective,
        linear_solver_cls=linear_solver_cls,
        max_iterations=max_iterations,
    )
    assert objective._cost_functions_iterable is not None  # vectorization is on
    assert isinstance(optimizer.linear_solver, linear_solver_cls)
    theseus_layer = th.TheseusLayer(optimizer)
    return theseus_layer


def get_average_sample_cost(
    x_samples, layer_to_learn, cost_weight_param_name, cost_weight_fn
):
    cost_opt = None
    n_samples = x_samples.shape[-1]
    for sidx in range(0, n_samples):
        input_values_opt = {
            "coefficients": x_samples[:, :, sidx],
            cost_weight_param_name: cost_weight_fn(),
        }
        layer_to_learn.objective.update(input_values_opt)
        if cost_opt is not None:
            cost_opt = cost_opt + torch.sum(layer_to_learn.objective.error(), dim=1)
        else:
            cost_opt = torch.sum(layer_to_learn.objective.error(), dim=1)
    cost_opt = cost_opt / n_samples

    return cost_opt


def test_layer_solver_constructor():
    dummy = torch.ones(1, 1)
    for linear_solver_cls in [th.LUDenseSolver, th.CholeskyDenseSolver]:
        layer = create_qf_theseus_layer(
            dummy, dummy, linear_solver_cls=linear_solver_cls
        )
        assert isinstance(
            layer.optimizer.linear_solver.linearization, th.DenseLinearization
        )
        assert isinstance(layer.optimizer.linear_solver, linear_solver_cls)
        assert isinstance(layer.optimizer, th.GaussNewton)


def _run_optimizer_test(
    nonlinear_optimizer_cls,
    linear_solver_cls,
    optimizer_kwargs,
    cost_weight_model,
    use_learnable_error=False,
    verbose=True,
    learning_method="default",
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"_run_test_for: {device}")
    print(
        f"testing for optimizer {nonlinear_optimizer_cls.__name__}, "
        f"cost weight modeled as {cost_weight_model}, "
        f"linear solver {linear_solver_cls.__name__}"
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    torch.manual_seed(0)  # fix global seed for mlp

    # Create the dataset to fit, model(x) is the true data generation process
    batch_size = 16
    num_points = 10
    xs = torch.linspace(0, 10, num_points).repeat(batch_size, 1).to(device)
    xs += 0.1 * torch.randn(batch_size, num_points, generator=rng, device=device)

    ys = model(xs, torch.ones(batch_size, 2, device=device))
    # Shift the y values a bit so there is no perfect fit and changing the
    # cost weight results in a different parameter fit
    fake_noise = torch.logspace(-4, 4, num_points, base=math.e).unsqueeze(0).to(device)
    ys -= fake_noise

    # First we create a quadratic fit problem with unit cost weight to see what
    # its solution is and use this solution as the target
    layer_ref = create_qf_theseus_layer(
        xs,
        ys,
        nonlinear_optimizer_cls=nonlinear_optimizer_cls,
        linear_solver_cls=linear_solver_cls,
        use_learnable_error=use_learnable_error,
    )
    layer_ref.to(device)
    with torch.no_grad():
        input_values = {"coefficients": torch.ones(batch_size, 2, device=device) * 0.75}
        target_vars, _ = layer_ref.forward(
            input_values, optimizer_kwargs={**optimizer_kwargs, **{"verbose": verbose}}
        )

    # Now create another that starts with a random cost weight and use backpropagation to
    # find the cost weight whose solution matches the above target
    # To do this, we create a diagonal cost weight with an auxiliary variable called
    # "cost_weight_values", which will get updated by the forward method of Theseus
    # layer.

    # Note: interestingly, if we pass a torch.Parameter parameter as the data to the
    # auxiliary variable of the cost weight, we don't even
    # need to pass updated values through "objective.update". I'm doing it this way
    # to check that update works properly
    cost_weight = th.DiagonalCostWeight(
        th.Variable(torch.empty(num_points), name="cost_weight_values")
    )

    # Here we create the outer loop models and optimizers for the cost weight
    if cost_weight_model == "softmax":

        cost_weight_params = nn.Parameter(
            torch.randn(num_points, generator=rng, device=device)
        )

        def cost_weight_fn():
            return F.softmax(cost_weight_params, dim=0).view(1, -1)

        optimizer = torch.optim.Adam([cost_weight_params], lr=0.075)

    elif cost_weight_model == "mlp":
        mlp = thutils.build_mlp(num_points, 20, num_points, 2).to(device)
        dummy_input = torch.ones(1, num_points, device=device)

        def cost_weight_fn():
            return mlp(dummy_input)

        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.075)

    layer_to_learn = create_qf_theseus_layer(
        xs,
        ys,
        cost_weight=cost_weight,
        nonlinear_optimizer_cls=nonlinear_optimizer_cls,
        linear_solver_cls=linear_solver_cls,
        use_learnable_error=use_learnable_error,
    )
    layer_to_learn.to(device)

    # Check the initial solution quality to check how much has loss improved later

    # When using learnable error function, we don't update the cost weight directly but do it
    # through the parameters of the learnable error
    cost_weight_param_name = (
        "learnable_err_param" if use_learnable_error else "cost_weight_values"
    )
    input_values = {
        "coefficients": torch.ones(batch_size, 2, device=device) * 0.75,
        cost_weight_param_name: cost_weight_fn(),
    }

    with torch.no_grad():
        pred_vars, info = layer_to_learn.forward(
            input_values, optimizer_kwargs=optimizer_kwargs
        )
        loss0 = F.mse_loss(
            pred_vars["coefficients"], target_vars["coefficients"]
        ).item()
        assert not (
            (info.status == th.NonlinearOptimizerStatus.START)
            | (info.status == th.NonlinearOptimizerStatus.FAIL)
        ).all()

    print("Initial loss: ", loss0)
    # --------- Learning happens here ---------#
    solved = False
    for i in range(200):
        optimizer.zero_grad()
        input_values = {
            "coefficients": torch.ones(batch_size, 2, device=device) * 0.75,
            cost_weight_param_name: cost_weight_fn(),
        }
        pred_vars, info = layer_to_learn.forward(
            input_values, optimizer_kwargs={**optimizer_kwargs, **{"verbose": verbose}}
        )
        assert not (
            (info.status == th.NonlinearOptimizerStatus.START)
            | (info.status == th.NonlinearOptimizerStatus.FAIL)
        ).all()

        mse_loss = F.mse_loss(pred_vars["coefficients"], target_vars["coefficients"])

        if learning_method == "leo":
            # groundtruth cost
            x_gt = target_vars["coefficients"]
            input_values_gt = {
                "coefficients": x_gt,
                cost_weight_param_name: cost_weight_fn(),
            }
            layer_to_learn.objective.update(input_values_gt)
            cost_gt = torch.sum(layer_to_learn.objective.error(), dim=1)

            # optimizer cost
            x_opt = pred_vars["coefficients"].detach()
            x_samples = layer_to_learn.compute_samples(
                layer_to_learn.optimizer.linear_solver, n_samples=10, temperature=1.0
            )  # batch_size x n_vars x n_samples
            if x_samples is None:  # use mean solution
                x_samples = x_opt.reshape(x_opt.shape[0], -1).unsqueeze(
                    -1
                )  # batch_size x n_vars x n_samples
            cost_opt = get_average_sample_cost(
                x_samples, layer_to_learn, cost_weight_param_name, cost_weight_fn
            )

            # loss value
            l2_reg = F.mse_loss(
                cost_weight_fn(), torch.zeros((1, num_points), device=device)
            )
            loss = (cost_gt - cost_opt) + 10.0 * l2_reg
            loss = torch.mean(loss, dim=0)
        else:
            loss = mse_loss

        loss.backward()
        optimizer.step()

        print(i, mse_loss.item())
        if mse_loss.item() / loss0 < 5e-3:
            solved = True
            break
    assert solved


def test_backward_gauss_newton():
    for use_learnable_error in [True, False]:
        for linear_solver_cls in [th.CholeskyDenseSolver, th.LUDenseSolver]:
            for cost_weight_model in ["softmax", "mlp"]:
                _run_optimizer_test(
                    th.GaussNewton,
                    linear_solver_cls,
                    {},
                    cost_weight_model,
                    use_learnable_error=use_learnable_error,
                )


def test_backward_gauss_newton_choleskysparse():
    for use_learnable_error in [True, False]:
        for cost_weight_model in ["softmax", "mlp"]:
            _run_optimizer_test(
                th.GaussNewton,
                th.CholmodSparseSolver,
                {},
                cost_weight_model,
                use_learnable_error=use_learnable_error,
            )


def test_backward_levenberg_marquardt():
    for use_learnable_error in [True, False]:
        for linear_solver_cls in [th.CholeskyDenseSolver, th.LUDenseSolver]:
            for cost_weight_model in ["softmax", "mlp"]:
                _run_optimizer_test(
                    th.LevenbergMarquardt,
                    linear_solver_cls,
                    {"damping": 0.01},
                    cost_weight_model,
                    use_learnable_error=use_learnable_error,
                )


def test_backward_levenberg_marquardt_choleskysparse():
    for use_learnable_error in [True, False]:
        for cost_weight_model in ["softmax", "mlp"]:
            _run_optimizer_test(
                th.LevenbergMarquardt,
                th.CholmodSparseSolver,
                {"damping": 0.01, "ellipsoidal_damping": False},
                cost_weight_model,
                use_learnable_error=use_learnable_error,
            )


def test_backward_gauss_newton_leo():
    for use_learnable_error in [True, False]:
        for linear_solver_cls in [th.CholeskyDenseSolver, th.LUDenseSolver]:
            for cost_weight_model in ["mlp"]:
                _run_optimizer_test(
                    th.GaussNewton,
                    linear_solver_cls,
                    {},
                    cost_weight_model,
                    use_learnable_error=use_learnable_error,
                    learning_method="leo",
                )


def test_backward_levenberg_marquardt_leo():
    for use_learnable_error in [True, False]:
        for linear_solver_cls in [th.CholeskyDenseSolver, th.LUDenseSolver]:
            for cost_weight_model in ["mlp"]:
                _run_optimizer_test(
                    th.LevenbergMarquardt,
                    linear_solver_cls,
                    {"damping": 0.01},
                    cost_weight_model,
                    use_learnable_error=use_learnable_error,
                    learning_method="leo",
                )


def test_send_to_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"test_send_to_device: {device}")

    # Create the dataset to fit, model(x) is the true data generation process
    batch_size = 16
    num_points = 10
    xs = torch.linspace(0, 10, num_points).repeat(batch_size, 1)
    ys = model(xs, torch.ones(batch_size, 2))

    layer = create_qf_theseus_layer(xs, ys)
    input_values = {"coefficients": torch.ones(batch_size, 2, device=device) * 0.5}
    with torch.no_grad():
        if device != "cpu":
            with pytest.raises(RuntimeError):
                layer.forward(input_values)
            layer.to(device)
            output_values, _ = layer.forward(input_values)
            for k, v in output_values.items():
                assert v.device == input_values[k].device


def test_check_objective_consistency():
    objective, *_ = create_objective_with_mock_cost_functions()
    optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver)

    def _do_check(layer_, optimizer_):
        with pytest.raises(RuntimeError):
            layer_.forward({})
        with pytest.raises(RuntimeError):
            optimizer_.optimize()

    # Check for adding a factor
    new_cost = MockCostFunction(
        [MockVar(1, name="dummy")],
        [],
        MockCostWeight(MockVar(1, name="weight_aux")),
    )
    layer = TheseusLayer(optimizer)
    objective.add(new_cost)
    _do_check(layer, optimizer)

    # Now check erasing a factor
    objective, cost_functions, *_ = create_objective_with_mock_cost_functions()
    optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver)
    objective.erase(cost_functions[0].name)
    _do_check(layer, optimizer)


def test_pass_optimizer_kwargs():
    # Create the dataset to fit, model(x) is the true data generation process
    batch_size = 16
    num_points = 10
    xs = torch.linspace(0, 10, num_points).repeat(batch_size, 1)
    ys = model(xs, torch.ones(batch_size, 2))

    layer = create_qf_theseus_layer(
        xs,
        ys,
        nonlinear_optimizer_cls=th.GaussNewton,
        linear_solver_cls=th.CholmodSparseSolver,
    )
    layer.to("cpu")
    input_values = {"coefficients": torch.ones(batch_size, 2) * 0.5}
    for tbs in [True, False]:
        _, info = layer.forward(
            input_values, optimizer_kwargs={"track_best_solution": tbs}
        )
        if tbs:
            assert (
                isinstance(info.best_solution, dict)
                and "coefficients" in info.best_solution
            )
        else:
            assert info.best_solution is None

    # Pass invalid backward mode to trigger exception
    with pytest.raises(ValueError):
        layer.forward(input_values, optimizer_kwargs={"backward_mode": -1})

    # Now test that compute_delta() args passed correctly
    # Path compute_delta() to receive args we control
    def _mock_compute_delta(cls, fake_arg=None, **kwargs):
        if fake_arg is not None:
            raise ValueError
        return layer.optimizer.linear_solver.solve()

    with mock.patch.object(th.GaussNewton, "compute_delta", _mock_compute_delta):
        layer_2 = create_qf_theseus_layer(xs, ys)
        layer_2.forward(input_values)
        # If fake_arg is passed correctly, the mock of compute_delta will trigger
        with pytest.raises(ValueError):
            layer_2.forward(input_values, {"fake_arg": True})


def test_no_layer_kwargs():
    # Create the dataset to fit, model(x) is the true data generation process
    batch_size = 16
    num_points = 10
    xs = torch.linspace(0, 10, num_points).repeat(batch_size, 1)
    ys = model(xs, torch.ones(batch_size, 2))

    layer = create_qf_theseus_layer(
        xs,
        ys,
        nonlinear_optimizer_cls=th.GaussNewton,
        linear_solver_cls=th.CholmodSparseSolver,
    )
    layer.to("cpu")
    input_values = {"coefficients": torch.ones(batch_size, 2) * 0.5}

    # Trying a few variations of aux_vars. In general, no kwargs should be accepted
    # beyong input_data and optimization_kwargs, but I'm not sure how to test for this
    with pytest.raises(TypeError):
        layer.forward(input_values, aux_vars=None)

    with pytest.raises(TypeError):
        layer.forward(input_values, aux_variables=None)

    with pytest.raises(TypeError):
        layer.forward(input_values, auxiliary_vars=None)
