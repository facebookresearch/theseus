# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

from theseus.core import (
    CostFunction,
    CostWeight,
    Objective,
    ScaleCostWeight,
    Variable,
    Vectorize,
)
from theseus.constants import __FROM_THESEUS_LAYER_TOKEN__
from theseus.geometry import LieGroup, Manifold
from theseus.optimizer import Optimizer, OptimizerInfo
from theseus.optimizer.linear import LinearSolver
from theseus.optimizer.nonlinear import BackwardMode, GaussNewton


class TheseusLayer(nn.Module):
    def __init__(
        self,
        optimizer: Optimizer,
        vectorize: bool = True,
        empty_cuda_cache: bool = False,
    ):
        super().__init__()
        self.objective = optimizer.objective
        if vectorize and not self.objective.vectorized:
            Vectorize(self.objective, empty_cuda_cache=empty_cuda_cache)
        self.optimizer = optimizer
        self._objectives_version = optimizer.objective.current_version
        self._dlm_bwd_objective = None
        self._dlm_bwd_optimizer = None

    def forward(
        self,
        input_tensors: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], OptimizerInfo]:
        if self._objectives_version != self.objective.current_version:
            raise RuntimeError(
                "The objective was modified after the layer's construction, which is "
                "currently not supported."
            )
        optimizer_kwargs = optimizer_kwargs or {}
        # Defaults to "unroll" to avoid error, we only care to see if it's not dlm.
        backward_mode = BackwardMode.resolve(
            optimizer_kwargs.get("backward_mode", "unroll")
        )
        if backward_mode == BackwardMode.DLM:
            dlm_epsilon = optimizer_kwargs.get(
                TheseusLayerDLMForward._DLM_EPSILON_STR, 1e-2
            )
            if not isinstance(dlm_epsilon, float):
                raise ValueError(
                    f"{TheseusLayerDLMForward._DLM_EPSILON_STR} must be a float "
                    f"but {type(dlm_epsilon)} was given."
                )

            if self._dlm_bwd_objective is None:
                _obj, _opt = _instantiate_dlm_bwd_objective(self.objective)
                _obj.to(self.device)
                self._dlm_bwd_objective = _obj
                self._dlm_bwd_optimizer = _opt

            # Tensors cannot be passed inside containers, else we run into memory leaks.
            input_keys, input_vals = zip(*input_tensors.items())
            differentiable_tensors = [t for t in input_vals if t.requires_grad]

            *vars, info = TheseusLayerDLMForward.apply(
                self.objective,
                self.optimizer,
                optimizer_kwargs,
                self._dlm_bwd_objective,
                self._dlm_bwd_optimizer,
                dlm_epsilon,
                len(input_keys),
                *input_keys,
                *input_vals,
                *differentiable_tensors,
            )
        else:
            vars, info = _forward(
                self.objective, self.optimizer, optimizer_kwargs, input_tensors
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
        delta_samples = (torch.linalg.solve_triangular(sqrt_AtA, y, upper=True)) + (
            delta.unsqueeze(-1)
        ).repeat(1, 1, n_samples)

        x_samples = torch.zeros((batch_size, n_vars, n_samples), device=delta.device)
        for sidx in range(0, n_samples):
            var_idx = 0
            for var in linear_solver.linearization.ordering:
                new_var = var.retract(
                    delta_samples[:, var_idx : var_idx + var.dof(), sidx]
                )
                x_samples[:, var_idx : var_idx + var.dof(), sidx] = new_var.tensor
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


def _forward(
    objective: Objective,
    optimizer: Optimizer,
    optimizer_kwargs: Dict[str, Any],
    input_tensors: Dict[str, torch.Tensor],
):
    objective.update(input_tensors)
    optimizer_kwargs[__FROM_THESEUS_LAYER_TOKEN__] = True
    info = optimizer.optimize(**optimizer_kwargs)
    vars = [var.tensor for var in objective.optim_vars.values()]
    return vars, info


class TheseusLayerDLMForward(torch.autograd.Function):
    """
    Functionally the same as the forward method in a TheseusLayer
    but computes the direct loss minimization in the backward pass.
    """

    _DLM_EPSILON_STR = "dlm_epsilon"
    _GRAD_SUFFIX = "_grad"

    @staticmethod
    def forward(
        ctx,
        objective,
        optimizer,
        optimizer_kwargs,
        bwd_objective,
        bwd_optimizer,
        epsilon,
        n,
        *inputs,
    ):
        input_keys = inputs[:n]
        input_vals = inputs[n : 2 * n]
        differentiable_tensors = inputs[2 * n :]
        ctx.n = n
        ctx.k = len(differentiable_tensors)

        inputs = dict(zip(input_keys, input_vals))
        ctx.input_keys = input_keys

        optim_tensors, info = _forward(objective, optimizer, optimizer_kwargs, inputs)

        # Skip computation if there are no differentiable inputs.
        if ctx.k > 0:
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
            ctx.save_for_backward(
                *input_vals, *grad_sol, *differentiable_tensors, *optim_tensors
            )
        return (*optim_tensors, info)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        n, k = ctx.n, ctx.k
        saved_tensors = ctx.saved_tensors
        input_vals = saved_tensors[:n]
        grad_sol = saved_tensors[n : n + k]
        differentiable_tensors = saved_tensors[n + k : n + k + k]
        optim_tensors = saved_tensors[n + k + k :]
        grad_outputs = grad_outputs[:-1]

        bwd_objective = ctx.bwd_objective
        bwd_optimizer = ctx.bwd_optimizer
        epsilon = ctx.epsilon
        input_keys = ctx.input_keys

        # Update the optim vars to their solutions.
        bwd_data = dict(zip(input_keys, input_vals))
        for k, v in zip(bwd_objective.optim_vars.keys(), optim_tensors):
            bwd_data[k] = v.detach()

        # Add in gradient values.
        grad_data = {
            TheseusLayerDLMForward._DLM_EPSILON_STR: torch.tensor(epsilon)
            .to(grad_outputs[0])
            .reshape(1, 1)
        }
        for i, name in enumerate(bwd_objective.optim_vars.keys()):
            grad_data[name + TheseusLayerDLMForward._GRAD_SUFFIX] = grad_outputs[i]
        bwd_data.update(grad_data)

        # Solve backward objective.
        bwd_objective.update(bwd_data)
        with torch.no_grad():
            bwd_optimizer.linear_solver.linearization.linearize()
            delta = bwd_optimizer.linear_solver.solve()
            bwd_optimizer.objective.retract_optim_vars(
                delta, bwd_optimizer.linear_solver.linearization.ordering
            )

        # Compute gradients.
        with torch.enable_grad():
            grad_perturbed = torch.autograd.grad(
                bwd_objective.error_squared_norm().sum(),
                differentiable_tensors,
                allow_unused=True,
            )

        nones = [None] * (ctx.n * 2)
        grads = [
            (gs - gp) / epsilon if gs is not None else None
            for gs, gp in zip(grad_sol, grad_perturbed)
        ]
        return (None, None, None, None, None, None, None, *nones, *grads)


class _DLMPerturbation(CostFunction):
    def __init__(
        self,
        var: Manifold,
        epsilon: Variable,
        grad: Variable,
        cost_weight: CostWeight,
        name: Optional[str] = None,
    ):
        if not isinstance(var, LieGroup):
            raise ValueError(
                f"DLM requires LieGroup-type variables, but "
                f"{var.name} has type {var.__class__.__name__}"
            )
        super().__init__(cost_weight, name=name)
        assert epsilon.ndim == 2 and epsilon.shape[1] == 1
        self.var = var
        self.epsilon = epsilon
        self.grad = grad
        self.register_optim_var("var")
        self.register_aux_vars(["epsilon", "grad"])

    def error(self) -> torch.Tensor:
        err = (
            self.epsilon.tensor.view((-1,) + (1,) * (self.var.ndim - 1))
            * self.var.tensor
            - 0.5 * self.grad.tensor
        )
        return err.flatten(start_dim=1)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        d = self.dim()
        aux = (
            torch.eye(d, dtype=self.epsilon.dtype, device=self.epsilon.device)
            .unsqueeze(0)
            .expand(self.var.shape[0], d, d)
        )
        euclidean_grad_flat = self.epsilon.tensor.view(-1, 1, 1) * aux
        euclidean_grad = euclidean_grad_flat.unflatten(2, self.var.shape[1:])
        return [self.var.project(euclidean_grad, is_sparse=True)], self.error()

    def dim(self) -> int:
        return int(np.prod(self.var.tensor.shape[1:]))

    def _copy_impl(self, new_name: Optional[str] = None) -> "CostFunction":
        return _DLMPerturbation(
            self.var.copy(),
            self.epsilon.copy(),
            self.grad.copy(),
            self.weight.copy(),
            name=new_name,
        )


def _instantiate_dlm_bwd_objective(objective: Objective):
    bwd_objective = objective.copy()
    epsilon_var = Variable(
        torch.ones(1, 1, dtype=bwd_objective.dtype, device=bwd_objective.device),
        name=TheseusLayerDLMForward._DLM_EPSILON_STR,
    )
    unit_weight = ScaleCostWeight(1.0)
    unit_weight.to(dtype=objective.dtype, device=objective.device)
    for name, var in bwd_objective.optim_vars.items():
        grad_var = Variable(
            torch.zeros_like(var.tensor),
            name=name + TheseusLayerDLMForward._GRAD_SUFFIX,
        )
        bwd_objective.add(
            _DLMPerturbation(
                var, epsilon_var, grad_var, unit_weight, name="dlm_perturbation" + name
            )
        )

    bwd_optimizer = GaussNewton(
        bwd_objective,
        max_iterations=1,
        step_size=1.0,
    )
    return bwd_objective, bwd_optimizer
