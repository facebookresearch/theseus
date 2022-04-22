# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

from functools import partial

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from theseus.core import Variable
from theseus.core.cost_function import AutoDiffCostFunction
from theseus.optimizer import Optimizer, OptimizerInfo
from theseus.optimizer.linear import LinearSolver
from theseus.optimizer.nonlinear import GaussNewton
from theseus.optimizer.nonlinear import BackwardMode


class TheseusLayer(nn.Module):
    def __init__(
        self,
        optimizer: Optimizer,
    ):
        super().__init__()
        self.objective = optimizer.objective
        self.optimizer = optimizer
        self._objectives_version = optimizer.objective.current_version

    def forward(
        self,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], OptimizerInfo]:
        if self._objectives_version != self.objective.current_version:
            raise RuntimeError(
                "The objective was modified after the layer's construction, which is "
                "currently not supported."
            )
        optimizer_kwargs = optimizer_kwargs or {}
        backward_mode = optimizer_kwargs.get("backward_mode", None)
        dlm_epsilon = optimizer_kwargs.get("dlm_epsilon", 1e-2)
        if backward_mode == BackwardMode.DLM:
            # TODO: instantiate self.bwd_objective here.
            names = set(self.objective.aux_vars.keys()).intersection(input_data.keys())
            tensors = [input_data[n] for n in names]
            *vars, info = TheseusLayerDLMForward.apply(
                self.objective, self.optimizer, optimizer_kwargs, input_data, dlm_epsilon, *tensors
            )
        else:
            vars, info = _forward(self.objective, self.optimizer, optimizer_kwargs, input_data)
        values = dict(zip(self.objective.optim_vars.keys(), vars))
        return values, info

    def compute_samples(
        self,
        linear_solver: LinearSolver = None,
        n_samples: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        # When samples are not available, return None. This makes the outer learning loop default
        # to a perceptron loss using the mean trajectory solution from the optimizer.
        if linear_solver is None:
            return None

        # Sampling from multivariate normal using a Cholesky decomposition of AtA,
        # http://www.statsathome.com/2018/10/19/sampling-from-multivariate-normal-precision-and-covariance-parameterizations/
        delta = linear_solver.solve()
        AtA = linear_solver.linearization.hessian_approx() / temperature
        sqrt_AtA = torch.linalg.cholesky(AtA).permute(0, 2, 1)

        batch_size, n_vars = delta.shape
        y = torch.normal(
            mean=torch.zeros((n_vars, n_samples), device=delta.device),
            std=torch.ones((n_vars, n_samples), device=delta.device),
        )
        delta_samples = (torch.triangular_solve(y, sqrt_AtA).solution) + (
            delta.unsqueeze(-1)
        ).repeat(1, 1, n_samples)

        x_samples = torch.zeros((batch_size, n_vars, n_samples), device=delta.device)
        for sidx in range(0, n_samples):
            var_idx = 0
            for var in linear_solver.linearization.ordering:
                new_var = var.retract(
                    delta_samples[:, var_idx : var_idx + var.dof(), sidx]
                )
                x_samples[:, var_idx : var_idx + var.dof(), sidx] = new_var.data
                var_idx = var_idx + var.dof()

        return x_samples

    # Applies to() with given args to all tensors in the objective
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.objective.to(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.objective.device

    @property
    def dtype(self) -> torch.dtype:
        return self.objective.dtype


def _forward(objective, optimizer, optimizer_kwargs, input_data):
    objective.update(input_data)
    info = optimizer.optimize(**optimizer_kwargs)
    vars = [var.data for var in objective.optim_vars.values()]
    return vars, info


class TheseusLayerDLMForward(torch.autograd.Function):
    """
    Functionally the same as the forward method in a TheseusLayer
    but computes the direct loss minimization in the backward pass.
    """

    @staticmethod
    def forward(ctx, objective, optimizer, optimizer_kwargs, input_data, epsilon, *params):
        optim_vars, info = _forward(objective, optimizer, optimizer_kwargs, input_data)

        ctx.input_data = input_data.copy()
        ctx.objective = objective
        ctx.epsilon = epsilon

        # Ideally we compute this in the backward function, but if we try to do that,
        # it ends up in an infinite loop because it depends on the outputs of this function.
        with torch.enable_grad():
            grad_sol = torch.autograd.grad(objective.error_squared_norm().sum(), params, retain_graph=True)

        ctx.save_for_backward(*params, *grad_sol, *optim_vars)
        ctx.n_params = len(params)
        return (*optim_vars, info)


    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        saved_tensors = ctx.saved_tensors
        params = saved_tensors[:ctx.n_params]
        grad_sol = saved_tensors[ctx.n_params:2 * ctx.n_params]
        optim_vars = saved_tensors[2 * ctx.n_params:]
        grad_outputs = grad_outputs[:-1]

        objective = ctx.objective
        epsilon = ctx.epsilon

        # Update the optim vars to their solutions.
        input_data = ctx.input_data
        values = dict(zip(objective.optim_vars.keys(), optim_vars))
        input_data.update(values)

        # Construct backward objective.
        bwd_objective = objective.copy()

        # Can we put all of this into a single cost function?
        for i, (name, var) in enumerate(bwd_objective.optim_vars.items()):
            grad_var = Variable(grad_outputs[i], name=name + "_grad")
            bwd_objective.add(AutoDiffCostFunction(
                [var],
                partial(_dlm_perturbation, epsilon=epsilon),
                1,
                aux_vars=[grad_var],
                name="DLM_perturbation_" + name,
            ))

        # Solve backward objective.
        bwd_objective.update(input_data)
        bwd_optimizer = GaussNewton(
            bwd_objective,
            max_iterations=1,
            step_size=1.0,
        )
        bwd_optimizer.optimize()

        # Compute gradients.
        with torch.enable_grad():
            grad_perturbed = torch.autograd.grad(bwd_objective.error_squared_norm().sum(), params, retain_graph=True)

        grads = [(gs - gp) / epsilon for gs, gp in zip(grad_sol, grad_perturbed)]
        return (None, None, None, None, None, *grads)


def _dlm_perturbation(optim_vars, aux_vars, epsilon):
    v = optim_vars[0]
    g = aux_vars[0]
    return epsilon * v.data - 0.5 * g.data
