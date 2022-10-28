import abc
import math
from multiprocessing.sharedctypes import Value
from typing import Any, Callable, Dict, NoReturn, Optional, Type, Union

import numpy as np
import torch
from lml import LML
from torch.distributions import Normal

import theseus.constants
from theseus.core.objective import Objective
from theseus.core.vectorizer import Vectorize
from theseus.optimizer import Optimizer, OptimizerInfo
from theseus.optimizer.variable_ordering import VariableOrdering

from .nonlinear_optimizer import (
    BackwardMode,
    NonlinearOptimizerInfo,
    NonlinearOptimizerParams,
    NonlinearOptimizerStatus,
)

EndIterCallbackType = Callable[
    ["DCem", NonlinearOptimizerInfo, torch.Tensor, int], NoReturn
]


class DCemSolver(abc.ABC):
    def __init__(
        self,
        objective: Objective,
        ordering: VariableOrdering = None,
        n_sample: int = 20,
        n_elite: int = 10,
        lb=None,
        ub=None,
        temp: float = 1.0,
        normalize: bool = False,
        lml_verbose: bool = False,
        lml_eps: float = 1e-3,
        **kwargs,
    ) -> None:
        self.objective = objective
        self.ordering = ordering
        self.n_samples = n_sample
        self.n_elite = n_elite
        self.lb = lb
        self.ub = ub
        self.temp = temp
        self.normalize = normalize
        self.tot_dof = sum([x.dof() for x in self.ordering])
        # self.sigma = {var.name: torch.ones(var.shape) for var in self.ordering}
        self.lml_eps = lml_eps

    def reinit_sigma(self):
        self.sigma = {var.name: torch.ones(var.shape) for var in self.ordering}

    def all_solve(self, num_iters):
        device = self.objective.device

        n_batch = self.ordering[0].shape[0]

        init_mu = torch.cat([var.tensor for var in self.ordering], dim=-1)
        init_sigma = torch.ones((n_batch, self.tot_dof), device=device)

        mu = init_mu.clone()
        sigma = init_sigma.clone()

        for itr in range(num_iters):
            X_samples = []
            for i in range(self.n_samples):
                sample = {}
                idx = 0
                for var in self.ordering:
                    sample[var.name] = Normal(
                        mu[:, slice(idx, idx + var.dof())],
                        sigma[:, slice(idx, idx + var.dof())],
                    ).rsample()
                    idx += var.dof()
                    assert sample[var.name].size() == var.shape
                    sample[var.name] = sample[var.name].contiguous()
                # print("sample", sample)
                X_samples.append(sample)

            fX = torch.stack(
                [
                    self.objective.error_squared_norm(X_samples[i])
                    for i in range(self.n_samples)
                ],
                dim=1,
            )

            # print("fx:", fX.shape)

            # assert fX.shape == (n_batch, self.n_samples)

            if self.temp is not None and self.temp < np.infty:
                if self.normalize:
                    fX_mu = fX.mean(dim=1).unsqueeze(1)
                    fX_sigma = fX.std(dim=1).unsqueeze(1)
                    _fX = (fX - fX_mu) / (fX_sigma + 1e-6)
                else:
                    _fX = fX

                if self.n_elite == 1:
                    # I = LML(N=n_elite, verbose=lml_verbose, eps=lml_eps)(-_fX*temp)
                    I = torch.softmax(-_fX * self.temp, dim=1)
                else:
                    I = LML(N=self.n_elite, verbose=False, eps=self.lml_eps)(
                        -_fX * self.temp
                    )
                I = I.unsqueeze(2)

            else:
                I_vals = fX.argsort(dim=1)[:, : self.n_elite]
                # TODO: A scatter would be more efficient here.
                I = torch.zeros(n_batch, self.n_samples, device=device)
                for j in range(n_batch):
                    for v in I_vals[j]:
                        I[j, v] = 1.0
                I = I.unsqueeze(2)
            # I.shape should be (n_batch, n_sample, 1)

            X = torch.zeros((n_batch, self.n_samples, self.tot_dof), device=device)

            for i in range(self.n_samples):
                sample = X_samples[i]
                idx = 0
                for var in self.ordering:
                    X[:, i, slice(idx, idx + var.dof())] = sample[var.name]
                    idx += var.dof()

            assert I.shape[:2] == X.shape[:2]
            # print("Samples:", X)
            X_I = I * X

            old_mu = mu.clone().detach()

            mu = torch.sum(X_I, dim=1) / self.n_elite
            sigma = ((I * (X - mu.unsqueeze(1)) ** 2).sum(dim=1) / self.n_elite).sqrt()

            assert sigma.shape == (n_batch, self.tot_dof)

            if (abs(mu - old_mu) < 1e-3).all():
                break

        new_dict = {}

        idx = 0
        for var in self.ordering:
            new_dict[var.name] = mu[:, slice(idx, idx + var.dof())]
            idx += var.dof()

        return new_dict

    def solve(self):
        device = self.objective.device

        n_batch = self.ordering[0].shape[0]
        mu = torch.zeros(
            (n_batch, sum([x.dof() for x in self.ordering])), device=device
        )

        idx = 0
        for var in self.ordering:
            mu[:, slice(idx, idx + var.dof())] = var.tensor
            idx += var.dof()

        # print("mu", mu)
        # print("sigma", self.sigma)

        X_samples = []
        for i in range(self.n_samples):
            sample = {}
            for var in self.ordering:
                sample[var.name] = (
                    Normal(var.tensor, self.sigma[var.name].to(device))
                    .rsample()
                    .to(device)
                )
                assert sample[var.name].size() == var.shape
                sample[var.name] = sample[var.name].contiguous()
                if self.lb is not None or self.ub is not None:
                    sample[var.name] = torch.clamp(sample[var.name], self.lb, self.ub)
            # print("sample", sample)
            X_samples.append(sample)

        # TODO: Check that self.objective.error_squared_norm(X_samples[i]).size() == (n_batch,)
        fX = torch.stack(
            [
                self.objective.error_squared_norm(X_samples[i])
                for i in range(self.n_samples)
            ],
            dim=1,
        )
        # print("fx:", fX.shape)

        # assert fX.shape == (n_batch, self.n_samples)

        if self.temp is not None and self.temp < np.infty:
            if self.normalize:
                fX_mu = fX.mean(dim=1).unsqueeze(1)
                fX_sigma = fX.std(dim=1).unsqueeze(1)
                _fX = (fX - fX_mu) / (fX_sigma + 1e-6)
            else:
                _fX = fX

            if self.n_elite == 1:
                # I = LML(N=n_elite, verbose=lml_verbose, eps=lml_eps)(-_fX*temp)
                I = torch.softmax(-_fX * self.temp, dim=1)
            else:
                I = LML(N=self.n_elite, verbose=False, eps=self.lml_eps)(
                    -_fX * self.temp
                )
            I = I.unsqueeze(2)

        else:
            I_vals = fX.argsort(dim=1)[:, : self.n_elite]
            # TODO: A scatter would be more efficient here.
            I = torch.zeros(n_batch, self.n_samples, device=device)
            for j in range(n_batch):
                for v in I_vals[j]:
                    I[j, v] = 1.0
            I = I.unsqueeze(2)
        # I.shape should be (n_batch, n_sample, 1)

        X = torch.zeros((n_batch, self.n_samples, self.tot_dof), device=device)

        for i in range(self.n_samples):
            sample = X_samples[i]
            idx = 0
            for var in self.ordering:
                X[:, i, slice(idx, idx + var.dof())] = sample[var.name]
                idx += var.dof()

        assert I.shape[:2] == X.shape[:2]
        # print("Samples:", X)
        X_I = I * X
        # old_mu = mu.clone()
        mu = torch.sum(X_I, dim=1) / self.n_elite

        sigma = ((I * (X - mu.unsqueeze(1)) ** 2).sum(dim=1) / self.n_elite).sqrt()
        # print("sigma_updates", sigma)

        assert sigma.shape == (n_batch, self.tot_dof)

        new_dict = {}

        idx = 0
        for var in self.ordering:
            self.sigma[var.name] = sigma[:, slice(idx, idx + var.dof())]
            new_dict[var.name] = mu[:, slice(idx, idx + var.dof())]
            idx += var.dof()

        # not sure about the detach
        # return mu - old_mu

        return new_dict


class DCem(Optimizer):
    def __init__(
        self,
        objective: Objective,
        vectorize: bool = False,
        cem_solver: Optional[abc.ABC] = DCemSolver,
        max_iterations: int = 20,
        n_sample: int = 50,
        n_elite: int = 5,
        temp: float = 1.0,
        lb=None,
        ub=None,
        lml_verbose: bool = False,
        lml_eps: float = 1e-3,
        normalize: bool = True,
        iter_eps: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(objective, vectorize=vectorize, **kwargs)

        self.params = NonlinearOptimizerParams(
            iter_eps, iter_eps * 100, max_iterations, 1e-2
        )

        self.ordering = VariableOrdering(objective)

        if cem_solver is None:
            cem_solver = DCemSolver

        self.linear_solver = cem_solver(
            objective,
            self.ordering,
            n_sample,
            n_elite,
            lb,
            ub,
            temp,
            normalize,
            lml_verbose,
            lml_eps,
        )

    def _maybe_init_best_solution(
        self, do_init: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not do_init:
            return None
        solution_dict = {}
        for var in self.ordering:
            solution_dict[var.name] = var.tensor.detach().clone().cpu()
        return solution_dict

    def _init_info(
        self,
        track_best_solution: bool,
        track_err_history: bool,
        track_state_history: bool,
    ) -> NonlinearOptimizerInfo:
        with torch.no_grad():
            last_err = self.objective.error_squared_norm() / 2
        best_err = last_err.clone() if track_best_solution else None
        if track_err_history:
            err_history = (
                torch.ones(self.objective.batch_size, self.params.max_iterations + 1)
                * math.inf
            )
            assert last_err.grad_fn is None
            err_history[:, 0] = last_err.clone().cpu()
        else:
            err_history = None

        if track_state_history:
            state_history = {}
            for var in self.objective.optim_vars.values():
                state_history[var.name] = (
                    torch.ones(
                        self.objective.batch_size,
                        *var.shape[1:],
                        self.params.max_iterations + 1,
                    )
                    * math.inf
                )
                state_history[var.name][..., 0] = var.tensor.detach().clone().cpu()
        else:
            state_history = None

        return NonlinearOptimizerInfo(
            best_solution=self._maybe_init_best_solution(do_init=track_best_solution),
            last_err=last_err,
            best_err=best_err,
            status=np.array(
                [NonlinearOptimizerStatus.START] * self.objective.batch_size
            ),
            converged_iter=torch.zeros_like(last_err, dtype=torch.long),
            best_iter=torch.zeros_like(last_err, dtype=torch.long),
            err_history=err_history,
            state_history=state_history,
        )

    def _check_convergence(self, err: torch.Tensor, last_err: torch.Tensor):
        assert not torch.is_grad_enabled()
        if err.abs().mean() < theseus.constants.EPS:
            return torch.ones_like(err).bool()

        abs_error = (last_err - err).abs()
        rel_error = abs_error / last_err
        return (abs_error < self.params.abs_err_tolerance).logical_or(
            rel_error < self.params.rel_err_tolerance
        )

    def _update_state_history(self, iter_idx: int, info: NonlinearOptimizerInfo):
        for var in self.objective.optim_vars.values():
            info.state_history[var.name][..., iter_idx + 1] = (
                var.tensor.detach().clone().cpu()
            )

    def _update_info(
        self,
        info: NonlinearOptimizerInfo,
        current_iter: int,
        err: torch.Tensor,
        converged_indices: torch.Tensor,
    ):
        info.converged_iter += 1 - converged_indices.long()
        if info.err_history is not None:
            assert err.grad_fn is None
            info.err_history[:, current_iter + 1] = err.clone().cpu()
        if info.state_history is not None:
            self._update_state_history(current_iter, info)

        if info.best_solution is not None:
            # Only copy best solution if needed (None means track_best_solution=False)
            assert info.best_err is not None
            good_indices = err < info.best_err
            info.best_iter[good_indices] = current_iter
            for var in self.ordering:
                info.best_solution[var.name][good_indices] = (
                    var.tensor.detach().clone()[good_indices].cpu()
                )

            info.best_err = torch.minimum(info.best_err, err)

        converged_indices = self._check_convergence(err, info.last_err)
        info.status[
            np.array(converged_indices.detach().cpu())
        ] = NonlinearOptimizerStatus.CONVERGED

    def _optimize_loop(
        self,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        **kwargs,
    ) -> int:

        mu = self.linear_solver.all_solve(num_iter)
        self.objective.update(mu)
        with torch.no_grad():
            info.best_solution = mu
        return

        converged_indices = torch.zeros_like(info.last_err).bool()
        iters_done = 0
        for it_ in range(num_iter):
            iters_done += 1
            try:
                mu = self.linear_solver.solve()
            except RuntimeError as error:
                raise RuntimeError(f"There is an error in update {error}")

            self.objective.update(mu)

            # check for convergence
            with torch.no_grad():
                err = self.objective.error_squared_norm() / 2
                self._update_info(info, it_, err, converged_indices)
                if verbose:
                    print(
                        f"Nonlinear optimizer. Iteration: {it_+1}. "
                        f"Error: {err.mean().item()}"
                    )
                converged_indices = self._check_convergence(err, info.last_err)
                info.status[
                    np.array(converged_indices.cpu().numpy())
                ] = NonlinearOptimizerStatus.CONVERGED
                if converged_indices.all():
                    break  # nothing else will happen at this point
                info.last_err = err

                if end_iter_callback is not None:
                    end_iter_callback(self, info, mu, it_)

        info.status[
            info.status == NonlinearOptimizerStatus.START
        ] = NonlinearOptimizerStatus.MAX_ITERATIONS

        return iters_done

    def _optimize_impl(
        self,
        track_best_solution: bool = True,
        track_err_history: bool = False,
        track_state_history: bool = True,
        verbose: bool = False,
        backward_mode: Union[str, BackwardMode] = BackwardMode.UNROLL,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        **kwargs,
    ) -> OptimizerInfo:
        backward_mode = BackwardMode.resolve(backward_mode)
        # self.linear_solver.reinit_sigma()

        with torch.no_grad():
            info = self._init_info(
                track_best_solution, track_err_history, track_state_history
            )

        if verbose:
            print(
                f"DCEM optimizer. Iteration: 0. "
                f"Error: {info.last_err.mean().item()}"
            )

        if backward_mode in [BackwardMode.UNROLL, BackwardMode.DLM]:
            self._optimize_loop(
                start_iter=0,
                num_iter=self.params.max_iterations,
                info=info,
                verbose=verbose,
                truncated_grad_loop=False,
                end_iter_callback=end_iter_callback,
                **kwargs,
            )
            # If didn't coverge, remove misleading converged_iter value
            info.converged_iter[
                info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
            ] = -1
            return info

        else:
            raise ValueError("Use Unroll as backward mode for now")
