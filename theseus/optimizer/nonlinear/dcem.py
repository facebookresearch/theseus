# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List, Dict

import numpy as np
import torch
from torch.distributions import Normal

from theseus.third_party.lml import LML
from theseus.core.objective import Objective
from theseus.optimizer import OptimizerInfo
from theseus.optimizer.variable_ordering import VariableOrdering

from .nonlinear_optimizer import (
    NonlinearOptimizer,
    BackwardMode,
    NonlinearOptimizerInfo,
    NonlinearOptimizerStatus,
    EndIterCallbackType,
)


class DCEM(NonlinearOptimizer):
    """
    DCEM optimizer for nonlinear optimization using sampling based techniques.
    The optimizer can be really sensitive to hypermeter tuning. Here are few tuning
    hints:
    1. If have to lower the max_iterations, then increase the n_sample.
    2. The higher the n_sample, the slowly with variance of samples will decrease.
    3. The higher the n_sample, more the chances of optimum being in the elite set.
    4. The higher the n_elite, the slower is convergence, but more accurate it might
       be, but would need more iterations. n_elite= 5 is good enough for most cases.
    """

    def __init__(
        self,
        objective: Objective,
        vectorize: bool = False,
        max_iterations: int = 50,
        n_sample: int = 100,
        n_elite: int = 5,
        temp: float = 1.0,
        init_sigma: Union[float, torch.Tensor] = 1.0,
        lb: float = None,
        ub: float = None,
        lml_verbose: bool = False,
        lml_eps: float = 1e-3,
        normalize: bool = True,
        abs_err_tolerance: float = 1e-6,
        rel_err_tolerance: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(
            objective,
            vectorize=vectorize,
            abs_err_tolerance=abs_err_tolerance,
            rel_err_tolerance=rel_err_tolerance,
            max_iterations=max_iterations,
            **kwargs,
        )

        self.objective = objective
        self.ordering = VariableOrdering(objective)
        self.n_samples = n_sample
        self.n_elite = n_elite
        self.lb = lb
        self.ub = ub
        self.temp = temp
        self.normalize = normalize
        self._tot_dof = sum([x.dof() for x in self.ordering])
        self.lml_eps = lml_eps
        self.lml_verbose = lml_verbose
        self.init_sigma = init_sigma

    def _mu_vec_to_dict(self, mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        idx = 0
        mu_dic = {}
        for var in self.ordering:
            mu_dic[var.name] = mu[:, slice(idx, idx + var.dof())]
            idx += var.dof()
        return mu_dic

    def reset_sigma(self, init_sigma: Union[float, torch.Tensor]) -> None:
        self.sigma = (
            torch.ones(
                (self.objective.batch_size, self._tot_dof), device=self.objective.device
            )
            * init_sigma
        )

    def _CEM_step(self):
        """
        Performs one iteration of CEM.
        Updates the self.sigma and return the new mu.
        """
        device = self.objective.device
        n_batch = self.ordering[0].shape[0]

        mu = torch.cat([var.tensor for var in self.ordering], dim=-1)

        X = Normal(mu, self.sigma).rsample((self.n_samples,))

        X_samples: List[Dict[str, torch.Tensor]] = []
        for sample in X:
            X_samples.append(self._mu_vec_to_dict(sample))

        fX = torch.stack(
            [self.objective.error_metric(X_samples[i]) for i in range(self.n_samples)],
            dim=1,
        )

        assert fX.shape == (n_batch, self.n_samples)

        if self.temp is not None and self.temp < np.infty:
            if self.normalize:
                fX_mu = fX.mean(dim=1).unsqueeze(1)
                fX_sigma = fX.std(dim=1).unsqueeze(1)
                _fX = (fX - fX_mu) / (fX_sigma + 1e-6)
            else:
                _fX = fX

            if self.n_elite == 1:
                # indexes = LML(N=n_elite, verbose=lml_verbose, eps=lml_eps)(-_fX*temp)
                indexes = torch.softmax(-_fX * self.temp, dim=1)
            else:
                indexes = LML(
                    N=self.n_elite, verbose=self.lml_verbose, eps=self.lml_eps
                )(-_fX * self.temp)
            indexes = indexes.unsqueeze(2)
            eps = 0

        else:
            indexes_vals = fX.argsort(dim=1)[:, : self.n_elite]
            # Scatter 1.0 to the indexes using indexes_vals
            indexes = torch.zeros(n_batch, self.n_samples, device=device).scatter_(
                1, indexes_vals, 1.0
            )
            indexes = indexes.unsqueeze(2)
            eps = 1e-10
        # indexes.shape should be (n_batch, n_sample, 1)

        X = X.transpose(0, 1)

        assert indexes.shape[:2] == X.shape[:2]

        X_I = indexes * X

        mu = torch.sum(X_I, dim=1) / self.n_elite
        self.sigma = (
            (indexes * (X - mu.unsqueeze(1)) ** 2).sum(dim=1) / self.n_elite
        ).sqrt() + eps  # adding eps to avoid sigma=0, which is happening when temp=None

        assert self.sigma.shape == (n_batch, self._tot_dof)

        return self._mu_vec_to_dict(mu)

    def _optimize_loop(
        self,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        **kwargs,
    ) -> int:
        converged_indices = torch.zeros_like(info.last_err).bool()
        iters_done = 0
        for it_ in range(num_iter):
            iters_done += 1
            try:
                mu = self._CEM_step()
            except RuntimeError as error:
                raise RuntimeError(f"There is an error in update {error}.")

            self.objective.update(mu)

            # check for convergence
            with torch.no_grad():
                err = self.objective.error_metric()
                self._update_info(info, it_, err, converged_indices)
                if verbose:
                    print(
                        f"Nonlinear optimizer. Iteration: {it_+1}. "
                        f"Error: {err.mean().item()} "
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
        track_best_solution: bool = False,
        track_err_history: bool = False,
        track_state_history: bool = False,
        verbose: bool = False,
        backward_mode: Union[str, BackwardMode] = BackwardMode.UNROLL,
        end_iter_callback: Optional[EndIterCallbackType] = None,
        **kwargs,
    ) -> OptimizerInfo:
        backward_mode = BackwardMode.resolve(backward_mode)
        init_sigma = kwargs.get("init_sigma", self.init_sigma)
        self.reset_sigma(init_sigma)

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
                num_iter=self.params.max_iterations,
                info=info,
                verbose=verbose,
                end_iter_callback=end_iter_callback,
                **kwargs,
            )
            # If didn't coverge, remove misleading converged_iter value
            info.converged_iter[
                info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
            ] = -1
            return info

        else:
            raise NotImplementedError(
                "DCEM currently only supports 'unroll' backward mode."
            )
