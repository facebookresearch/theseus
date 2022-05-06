# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from theseus.core import Variable
from theseus.core.cost_function import AutoDiffCostFunction
from theseus.optimizer import Optimizer, OptimizerInfo
from theseus.optimizer.linear import LinearSolver
from theseus.optimizer.nonlinear import BackwardMode, GaussNewton


class TheseusLayer(nn.Module):
    def __init__(
        self,
        optimizer: Optimizer,
    ):
        super().__init__()
        self.objective = optimizer.objective
        self.optimizer = optimizer
        self._objectives_version = optimizer.objective.current_version

        self._dlm_bwd_objective = None
        self._dlm_bwd_optimizer = None

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
        self.objective.setup(input_data)
        self.objective.update(input_data)
        optimizer_kwargs = optimizer_kwargs or {}
        backward_mode = optimizer_kwargs.get("backward_mode", None)
        dlm_epsilon = optimizer_kwargs.get(TheseusLayerDLMForward._dlm_epsilon, 1e-2)
        if backward_mode == BackwardMode.DLM:

            if self._dlm_bwd_objective is None:
                _obj, _opt = _instantiate_dlm_bwd_objective(self.objective)
                _obj.to(self.device)
                self._dlm_bwd_objective = _obj
                self._dlm_bwd_optimizer = _opt

            input_keys, input_vals = zip(*input_data.items())
            *vars, info = TheseusLayerDLMForward.apply(
                self.objective,
                self.optimizer,
                optimizer_kwargs,
                self._dlm_bwd_objective,
                self._dlm_bwd_optimizer,
                dlm_epsilon,
                *input_keys,
                *input_vals,
            )
        else:
            vars, info = _forward(
                self.objective, self.optimizer, optimizer_kwargs, input_data
            )
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

    _dlm_epsilon = "dlm_epsilon"
    _grad_suffix = "_grad"

    @staticmethod
    def forward(
        ctx,
        objective,
        optimizer,
        optimizer_kwargs,
        bwd_objective,
        bwd_optimizer,
        epsilon,
        *input_data,
    ):
        n = len(input_data) // 2
        input_keys = input_data[:n]
        input_vals = input_data[n:]
        ctx.n = n

        input_data = dict(zip(input_keys, input_vals))
        ctx.input_keys = input_keys

        differentiable_keys = [k for k in input_keys if input_data[k].requires_grad]
        differentiable_tensors = [v for v in input_vals if v.requires_grad]
        ctx.differentiable_keys = differentiable_keys

        optim_tensors, info = _forward(
            objective, optimizer, optimizer_kwargs, input_data
        )

        # Skip computation if there are no differentiable inputs.
        if len(differentiable_keys) > 0:
            ctx.bwd_objective = bwd_objective
            ctx.bwd_optimizer = bwd_optimizer
            ctx.epsilon = epsilon

            # Precompute and cache this.
            with torch.enable_grad():
                grad_sol = torch.autograd.grad(
                    objective.error_squared_norm().sum(),
                    differentiable_tensors,
                    allow_unused=True,
                )
            ctx.k = len(differentiable_keys)
            ctx.save_for_backward(*input_vals, *grad_sol, *optim_tensors)
        return (*optim_tensors, info)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        saved_tensors = ctx.saved_tensors
        input_vals = saved_tensors[: ctx.n]
        grad_sol = saved_tensors[ctx.n : ctx.n + ctx.k]
        optim_tensors = saved_tensors[ctx.n + ctx.k :]
        grad_outputs = grad_outputs[:-1]

        bwd_objective = ctx.bwd_objective
        bwd_optimizer = ctx.bwd_optimizer
        epsilon = ctx.epsilon
        input_keys = ctx.input_keys
        differentiable_keys = ctx.differentiable_keys

        # Update the optim vars to their solutions.
        bwd_data = dict(zip(input_keys, input_vals))
        sol_values = dict(zip(bwd_objective.optim_vars.keys(), optim_tensors))
        bwd_data.update(sol_values)

        # Add in gradient values.
        grad_data = {
            TheseusLayerDLMForward._dlm_epsilon: torch.tensor(epsilon)
            .to(grad_outputs[0])
            .reshape(1, 1)
        }
        for i, name in enumerate(bwd_objective.optim_vars.keys()):
            grad_data[name + TheseusLayerDLMForward._grad_suffix] = grad_outputs[i]
        bwd_data.update(grad_data)

        # Solve backward objective.
        bwd_objective.update(bwd_data)
        bwd_optimizer.optimize()

        differentiable_tensors = [bwd_data[k] for k in differentiable_keys]

        # Compute gradients.
        with torch.enable_grad():
            grad_perturbed = torch.autograd.grad(
                bwd_objective.error_squared_norm().sum(),
                differentiable_tensors,
                allow_unused=True,
            )

        grads = []
        i = 0
        for k in input_keys:
            if i < len(differentiable_keys) and k == differentiable_keys[i]:
                if grad_sol[i] is not None:
                    grads.append((grad_sol[i] - grad_perturbed[i]) / epsilon)
                else:
                    grads.append(None)
                i += 1
            else:
                grads.append(None)
        nones = [None] * len(grads)
        return (None, None, None, None, None, None, *nones, *grads)


def _dlm_perturbation(optim_vars, aux_vars):
    v = optim_vars[0]
    g = aux_vars[0]
    epsilon = aux_vars[1]
    return epsilon.data * v.data - 0.5 * g.data


def _instantiate_dlm_bwd_objective(objective):
    bwd_objective = objective.copy()
    epsilon_var = Variable(torch.ones(1, 1), name=TheseusLayerDLMForward._dlm_epsilon)
    for name, var in bwd_objective.optim_vars.items():
        grad_var = Variable(
            torch.zeros_like(var.data), name=name + TheseusLayerDLMForward._grad_suffix
        )
        bwd_objective.add(
            AutoDiffCostFunction(
                [var],
                _dlm_perturbation,
                var.shape[1],
                aux_vars=[grad_var, epsilon_var],
                name="dlm_perturbation_" + name,
                optim_var_names=["orig_var"],
                aux_var_names=["grad_var", "epsilon_var"],
            ),
            use_batches=True,
        )

    bwd_optimizer = GaussNewton(
        bwd_objective,
        max_iterations=1,
        step_size=1.0,
    )
    return bwd_objective, bwd_optimizer
