# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch

import theseus as th
import theseus.constants
from theseus.core import CostFunction, Objective
from theseus.geometry import Manifold
from theseus.optimizer import Linearization, Optimizer, VariableOrdering
from theseus.optimizer.nonlinear.nonlinear_optimizer import (
    BackwardMode,
    NonlinearOptimizerInfo,
    NonlinearOptimizerStatus,
)

"""
TODO
 - Parallelise factor to variable message comp
 - Benchmark speed
 - test jax implementation of message comp functions
 - add class for message schedule
 - damping for lie algebra vars
 - solving inverse problem to compute message mean
"""


"""
Utitily functions
"""


@dataclass
class GBPOptimizerParams:
    abs_err_tolerance: float
    rel_err_tolerance: float
    max_iterations: int

    def update(self, params_dict):
        for param, value in params_dict.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid nonlinear optimizer parameter {param}.")


# Stores variable beliefs that converge towards the marginals
class Gaussian:
    def __init__(self, variable: Manifold):
        self.name = variable.name + "_gaussian"
        self.mean = variable
        self.tot_dof = variable.dof()

        # tot_dof = 0
        # for v in variables:
        #     tot_dof += v.dof()
        # self.tot_dof = tot_dof

        self.precision = torch.zeros(
            self.mean.shape[0], self.tot_dof, self.tot_dof, dtype=variable.dtype
        )

    def dof(self) -> int:
        return self.tot_dof


class Marginals(Gaussian):
    pass


class Message(Gaussian):
    pass


class CostFunctionOrdering:
    def __init__(self, objective: Objective, default_order: bool = True):
        self.objective = objective
        self._cf_order: List[CostFunction] = []
        self._cf_name_to_index: Dict[str, int] = {}
        if default_order:
            self._compute_default_order(objective)

    def _compute_default_order(self, objective: Objective):
        assert not self._cf_order and not self._cf_name_to_index
        cur_idx = 0
        for cf_name, cf in objective.cost_functions.items():
            if cf_name in self._cf_name_to_index:
                continue
            self._cf_order.append(cf)
            self._cf_name_to_index[cf_name] = cur_idx
            cur_idx += 1

    def index_of(self, key: str) -> int:
        return self._cf_name_to_index[key]

    def __getitem__(self, index) -> CostFunction:
        return self._cf_order[index]

    def __iter__(self):
        return iter(self._cf_order)

    def append(self, cf: CostFunction):
        if cf in self._cf_order:
            raise ValueError(
                f"Cost Function {cf.name} has already been added to the order."
            )
        if cf.name not in self.objective.cost_functions:
            raise ValueError(
                f"Cost Function {cf.name} is not a cost function for the objective."
            )
        self._cf_order.append(cf)
        self._cf_name_to_index[cf.name] = len(self._cf_order) - 1

    def remove(self, cf: CostFunction):
        self._cf_order.remove(cf)
        del self._cf_name_to_index[cf.name]

    def extend(self, cfs: Sequence[CostFunction]):
        for cf in cfs:
            self.append(cf)

    @property
    def complete(self):
        return len(self._cf_order) == self.objective.size_variables()


"""
GBP functions
"""


# Compute the factor at current adjacent beliefs.
def compute_factor(cf, lie=True):
    J, error = cf.weighted_jacobians_error()
    J_stk = torch.cat(J, dim=-1)

    lam = torch.bmm(J_stk.transpose(-2, -1), J_stk)

    optim_vars_stk = torch.cat([v.data for v in cf.optim_vars], dim=-1)
    eta = -torch.matmul(J_stk.transpose(-2, -1), error.unsqueeze(-1))
    if lie is False:
        eta = eta + torch.matmul(lam, optim_vars_stk.unsqueeze(-1))
    eta = eta.squeeze(-1)

    return eta, lam


def pass_var_to_fac_messages(
    ftov_msgs_eta,
    ftov_msgs_lam,
    var_ix_for_edges,
    n_vars,
    max_dofs,
):
    belief_eta = torch.zeros(n_vars, max_dofs, dtype=ftov_msgs_eta.dtype)
    belief_lam = torch.zeros(n_vars, max_dofs, max_dofs, dtype=ftov_msgs_eta.dtype)

    belief_eta = belief_eta.index_add(0, var_ix_for_edges, ftov_msgs_eta)
    belief_lam = belief_lam.index_add(0, var_ix_for_edges, ftov_msgs_lam)

    vtof_msgs_eta = belief_eta[var_ix_for_edges] - ftov_msgs_eta
    vtof_msgs_lam = belief_lam[var_ix_for_edges] - ftov_msgs_lam

    return vtof_msgs_eta, vtof_msgs_lam, belief_eta, belief_lam


def pass_fac_to_var_messages(
    potentials_eta,
    potentials_lam,
    vtof_msgs_eta,
    vtof_msgs_lam,
    adj_var_dofs_nested: List[List],
):
    ftov_msgs_eta = torch.zeros_like(vtof_msgs_eta)
    ftov_msgs_lam = torch.zeros_like(vtof_msgs_lam)

    start = 0
    for i in range(len(adj_var_dofs_nested)):
        adj_var_dofs = adj_var_dofs_nested[i]
        num_optim_vars = len(adj_var_dofs)

        ftov_eta, ftov_lam = ftov_comp_mess(
            adj_var_dofs,
            potentials_eta[i],
            potentials_lam[i],
            vtof_msgs_eta[start : start + num_optim_vars],
            vtof_msgs_lam[start : start + num_optim_vars],
        )

        ftov_msgs_eta[start : start + num_optim_vars] = torch.cat(ftov_eta)
        ftov_msgs_lam[start : start + num_optim_vars] = torch.cat(ftov_lam)

        start += num_optim_vars

    return ftov_msgs_eta, ftov_msgs_lam


# Transforms message gaussian to tangent plane at var
# if return_mean is True, return the (mean, lam) else return (eta, lam).
# Generalises the local function by transforming the covariance as well as mean.
def local_gaussian(
    mess: Message,
    var: th.LieGroup,
    return_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # mean_tp is message mean in tangent space / plane at var
    mean_tp = var.local(mess.mean)

    jac: List[torch.Tensor] = []
    th.exp_map(var, mean_tp, jacobians=jac)

    # lam_tp is the precision matrix in the tangent plane
    lam_tp = torch.bmm(torch.bmm(jac[0].transpose(-1, -2), mess.precision), jac[0])

    if return_mean:
        return mean_tp, lam_tp

    else:
        eta_tp = torch.matmul(lam_tp, mean_tp.unsqueeze(-1)).squeeze(-1)
        return eta_tp, lam_tp


# Transforms Gaussian in the tangent plane at var to Gaussian where the mean
# is a group element and the precision matrix is defined in the tangent plane
# at the mean.
# Generalises the retract function by transforming the covariance as well as mean.
# out_gauss is the transformed Gaussian that is updated in place.
def retract_gaussian(
    mean_tp: torch.Tensor,
    prec_tp: torch.Tensor,
    var: th.LieGroup,
    out_gauss: Gaussian,
):
    mean = var.retract(mean_tp)

    jac: List[torch.Tensor] = []
    th.exp_map(var, mean_tp, jacobians=jac)
    inv_jac = torch.inverse(jac[0])
    precision = torch.bmm(torch.bmm(inv_jac.transpose(-1, -2), prec_tp), inv_jac)

    out_gauss.mean.update(mean.data)
    out_gauss.precision = precision


def pass_var_to_fac_messages_and_update_beliefs_lie(
    ftov_msgs,
    vtof_msgs,
    var_ordering,
    var_ix_for_edges,
):
    for i, var in enumerate(var_ordering):

        # Collect all incoming messages in the tangent space at the current belief
        taus = []  # message means
        lams_tp = []  # message lams
        for j, msg in enumerate(ftov_msgs):
            if var_ix_for_edges[j] == i:
                tau, lam_tp = local_gaussian(msg, var, return_mean=True)
                taus.append(tau[None, ...])
                lams_tp.append(lam_tp[None, ...])

        taus = torch.cat(taus)
        lams_tp = torch.cat(lams_tp)

        lam_tau = lams_tp.sum(dim=0)

        # Compute outgoing messages
        ix = 0
        for j, msg in enumerate(ftov_msgs):
            if var_ix_for_edges[j] == i:
                taus_inc = torch.cat((taus[:ix], taus[ix + 1 :]))
                lams_inc = torch.cat((lams_tp[:ix], lams_tp[ix + 1 :]))

                lam_a = lams_inc.sum(dim=0)
                if lam_a.count_nonzero() == 0:
                    vtof_msgs[j].mean.data[:] = 0.0
                    vtof_msgs[j].precision = lam_a
                else:
                    inv_lam_a = torch.inverse(lam_a)
                    sum_taus = torch.matmul(lams_inc, taus_inc.unsqueeze(-1)).sum(dim=0)
                    tau_a = torch.matmul(inv_lam_a, sum_taus).squeeze(-1)
                    retract_gaussian(tau_a, lam_a, var, vtof_msgs[j])
                ix += 1

        # update belief mean and variance
        # if no incoming messages then leave current belief unchanged
        if lam_tau.count_nonzero() != 0:
            inv_lam_tau = torch.inverse(lam_tau)
            sum_taus = torch.matmul(lams_tp, taus.unsqueeze(-1)).sum(dim=0)
            tau = torch.matmul(inv_lam_tau, sum_taus).squeeze(-1)

            belief = Gaussian(var)
            retract_gaussian(tau, lam_tau, var, belief)


def pass_fac_to_var_messages_lie(
    potentials_eta,
    potentials_lam,
    lin_points,
    vtof_msgs,
    ftov_msgs,
    adj_var_dofs_nested: List[List],
    damping: torch.Tensor,
):
    start = 0
    for i in range(len(adj_var_dofs_nested)):
        adj_var_dofs = adj_var_dofs_nested[i]
        num_optim_vars = len(adj_var_dofs)

        ftov_comp_mess_lie(
            potentials_eta[i],
            potentials_lam[i],
            lin_points[i],
            vtof_msgs[start : start + num_optim_vars],
            ftov_msgs[start : start + num_optim_vars],
            damping[start : start + num_optim_vars],
        )

        start += num_optim_vars


# Compute all outgoing messages from the factor.
def ftov_comp_mess(
    adj_var_dofs,
    potential_eta,
    potential_lam,
    vtof_msgs_eta,
    vtof_msgs_lam,
):
    num_optim_vars = len(adj_var_dofs)
    messages_eta, messages_lam = [], []

    sdim = 0
    for v in range(num_optim_vars):
        eta_factor = potential_eta.clone()[0]
        lam_factor = potential_lam.clone()[0]

        # Take product of factor with incoming messages
        start = 0
        for var in range(num_optim_vars):
            var_dofs = adj_var_dofs[var]
            if var != v:
                eta_mess = vtof_msgs_eta[var]
                lam_mess = vtof_msgs_lam[var]
                eta_factor[start : start + var_dofs] += eta_mess
                lam_factor[
                    start : start + var_dofs, start : start + var_dofs
                ] += lam_mess
            start += var_dofs

        # Divide up parameters of distribution
        dofs = adj_var_dofs[v]
        eo = eta_factor[sdim : sdim + dofs]
        eno = np.concatenate((eta_factor[:sdim], eta_factor[sdim + dofs :]))

        loo = lam_factor[sdim : sdim + dofs, sdim : sdim + dofs]
        lono = np.concatenate(
            (
                lam_factor[sdim : sdim + dofs, :sdim],
                lam_factor[sdim : sdim + dofs, sdim + dofs :],
            ),
            axis=1,
        )
        lnoo = np.concatenate(
            (
                lam_factor[:sdim, sdim : sdim + dofs],
                lam_factor[sdim + dofs :, sdim : sdim + dofs],
            ),
            axis=0,
        )
        lnono = np.concatenate(
            (
                np.concatenate(
                    (lam_factor[:sdim, :sdim], lam_factor[:sdim, sdim + dofs :]), axis=1
                ),
                np.concatenate(
                    (
                        lam_factor[sdim + dofs :, :sdim],
                        lam_factor[sdim + dofs :, sdim + dofs :],
                    ),
                    axis=1,
                ),
            ),
            axis=0,
        )

        new_message_lam = loo - lono @ np.linalg.inv(lnono) @ lnoo
        new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno

        messages_eta.append(new_message_eta[None, :])
        messages_lam.append(new_message_lam[None, :])

        sdim += dofs

    return messages_eta, messages_lam


# Compute all outgoing messages from the factor.
def ftov_comp_mess_lie(
    potential_eta,
    potential_lam,
    lin_points,
    vtof_msgs,
    ftov_msgs,
    damping,
):
    num_optim_vars = len(lin_points)
    new_messages = []

    sdim = 0
    for v in range(num_optim_vars):
        eta_factor = potential_eta.clone()[0]
        lam_factor = potential_lam.clone()[0]

        # Take product of factor with incoming messages.
        # Convert mesages to tangent space at linearisation point.
        start = 0
        for i in range(num_optim_vars):
            var_dofs = lin_points[i].dof()
            if i != v:
                eta_mess, lam_mess = local_gaussian(
                    vtof_msgs[i], lin_points[i], return_mean=False
                )
                eta_factor[start : start + var_dofs] += eta_mess[0]
                lam_factor[
                    start : start + var_dofs, start : start + var_dofs
                ] += lam_mess[0]
            start += var_dofs

        # Divide up parameters of distribution
        dofs = lin_points[v].dof()
        eo = eta_factor[sdim : sdim + dofs]
        eno = np.concatenate((eta_factor[:sdim], eta_factor[sdim + dofs :]))

        loo = lam_factor[sdim : sdim + dofs, sdim : sdim + dofs]
        lono = np.concatenate(
            (
                lam_factor[sdim : sdim + dofs, :sdim],
                lam_factor[sdim : sdim + dofs, sdim + dofs :],
            ),
            axis=1,
        )
        lnoo = np.concatenate(
            (
                lam_factor[:sdim, sdim : sdim + dofs],
                lam_factor[sdim + dofs :, sdim : sdim + dofs],
            ),
            axis=0,
        )
        lnono = np.concatenate(
            (
                np.concatenate(
                    (lam_factor[:sdim, :sdim], lam_factor[:sdim, sdim + dofs :]), axis=1
                ),
                np.concatenate(
                    (
                        lam_factor[sdim + dofs :, :sdim],
                        lam_factor[sdim + dofs :, sdim + dofs :],
                    ),
                    axis=1,
                ),
            ),
            axis=0,
        )

        new_mess_lam = loo - lono @ np.linalg.inv(lnono) @ lnoo
        new_mess_eta = eo - lono @ np.linalg.inv(lnono) @ eno

        # damping in tangent space at linearisation point
        # prev_mess_eta, prev_mess_lam = local_gaussian(
        #     vtof_msgs[v], lin_points[v], return_mean=False)
        # new_mess_eta = (1 - damping[v]) * new_mess_eta + damping[v] * prev_mess_eta[0]
        # new_mess_lam = (1 - damping[v]) * new_mess_lam + damping[v] * prev_mess_lam[0]

        if new_mess_lam.count_nonzero() == 0:
            new_mess = Gaussian(lin_points[v].copy())
            new_mess.mean.data[:] = 0.0
        else:
            new_mess_mean = torch.matmul(torch.inverse(new_mess_lam), new_mess_eta)
            new_mess_mean = new_mess_mean[None, ...]
            new_mess_lam = new_mess_lam[None, ...]

            new_mess = Gaussian(lin_points[v].copy())
            retract_gaussian(new_mess_mean, new_mess_lam, lin_points[v], new_mess)
        new_messages.append(new_mess)

        sdim += dofs

    # update messages
    for v in range(num_optim_vars):
        ftov_msgs[v].mean.update(new_messages[v].mean.data)
        ftov_msgs[v].precision = new_messages[v].precision

    return new_messages


# Follows notation from https://arxiv.org/pdf/2202.03314.pdf


class GaussianBeliefPropagation(Optimizer, abc.ABC):
    def __init__(
        self,
        objective: Objective,
        *args,
        linearization_cls: Optional[Type[Linearization]] = None,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        abs_err_tolerance: float = 1e-10,
        rel_err_tolerance: float = 1e-8,
        max_iterations: int = 20,
    ):
        super().__init__(objective)

        # ordering is required to identify which messages to send where
        self.ordering = VariableOrdering(objective, default_order=True)
        self.cf_ordering = CostFunctionOrdering(objective)

        self.schedule = None

        self.params = GBPOptimizerParams(
            abs_err_tolerance, rel_err_tolerance, max_iterations
        )

        self.n_edges = sum([cf.num_optim_vars() for cf in self.cf_ordering])
        self.max_dofs = max([var.dof() for var in self.ordering])

        # create arrays for indexing the messages
        var_ixs_nested = [
            [self.ordering.index_of(var.name) for var in cf.optim_vars]
            for cf in self.cf_ordering
        ]
        var_ixs = [item for sublist in var_ixs_nested for item in sublist]
        self.var_ix_for_edges = torch.tensor(var_ixs).long()

        self.adj_var_dofs_nested = [
            [var.shape[1] for var in cf.optim_vars] for cf in self.cf_ordering
        ]

        lie_groups = False
        for v in self.ordering:
            if isinstance(v, th.LieGroup) and not isinstance(v, th.Vector):
                lie_groups = True
        self.lie_groups = lie_groups
        print("lie groups:", self.lie_groups)

    """
    Copied and slightly modified from nonlinear optimizer class
    """

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def _check_convergence(self, err: torch.Tensor, last_err: torch.Tensor):
        assert not torch.is_grad_enabled()
        if err.abs().mean() < theseus.constants.EPS:
            return torch.ones_like(err).bool()

        abs_error = (last_err - err).abs()
        rel_error = abs_error / last_err
        return (abs_error < self.params.abs_err_tolerance).logical_or(
            rel_error < self.params.rel_err_tolerance
        )

    def _maybe_init_best_solution(
        self, do_init: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not do_init:
            return None
        solution_dict = {}
        for var in self.ordering:
            solution_dict[var.name] = var.data.detach().clone().cpu()
        return solution_dict

    def _init_info(
        self, track_best_solution: bool, track_err_history: bool, verbose: bool
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

        if info.best_solution is not None:
            # Only copy best solution if needed (None means track_best_solution=False)
            assert info.best_err is not None
            good_indices = err < info.best_err
            info.best_iter[good_indices] = current_iter
            for var in self.ordering:
                info.best_solution[var.name][good_indices] = (
                    var.data.detach().clone()[good_indices].cpu()
                )

            info.best_err = torch.minimum(info.best_err, err)

        converged_indices = self._check_convergence(err, info.last_err)
        info.status[
            np.array(converged_indices.detach().cpu())
        ] = NonlinearOptimizerStatus.CONVERGED

    # Modifies the (no grad) info in place to add data of grad loop info
    def _merge_infos(
        self,
        grad_loop_info: NonlinearOptimizerInfo,
        num_no_grad_iter: int,
        backward_num_iterations: int,
        info: NonlinearOptimizerInfo,
    ):
        # Concatenate error histories
        if info.err_history is not None:
            info.err_history[:, num_no_grad_iter:] = grad_loop_info.err_history[
                :, : backward_num_iterations + 1
            ]
        # Merge best solution and best error
        if info.best_solution is not None:
            best_solution = {}
            best_err_no_grad = info.best_err
            best_err_grad = grad_loop_info.best_err
            idx_no_grad = best_err_no_grad < best_err_grad
            best_err = torch.minimum(best_err_no_grad, best_err_grad)
            for var_name in info.best_solution:
                sol_no_grad = info.best_solution[var_name]
                sol_grad = grad_loop_info.best_solution[var_name]
                best_solution[var_name] = torch.where(
                    idx_no_grad, sol_no_grad, sol_grad
                )
            info.best_solution = best_solution
            info.best_err = best_err

        # Merge the converged status into the info from the detached loop,
        M = info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
        assert np.all(
            (grad_loop_info.status[M] == NonlinearOptimizerStatus.MAX_ITERATIONS)
            | (grad_loop_info.status[M] == NonlinearOptimizerStatus.CONVERGED)
        )
        info.status[M] = grad_loop_info.status[M]
        info.converged_iter[M] = (
            info.converged_iter[M] + grad_loop_info.converged_iter[M]
        )
        # If didn't coverge in either loop, remove misleading converged_iter value
        info.converged_iter[
            M & (grad_loop_info.status == NonlinearOptimizerStatus.MAX_ITERATIONS)
        ] = -1

    """
    GBP specific functions
    """

    # Linearizes factors at current belief if beliefs have deviated
    # from the linearization point by more than the threshold.
    def _linearize(
        self,
        potentials_eta,
        potentials_lam,
        lin_points,
        lp_dist_thresh: float = None,
        lie=False,
    ):
        do_lins = []
        for i, cf in enumerate(self.cf_ordering):

            do_lin = False
            if lp_dist_thresh is None:
                do_lin = True
            else:
                lp_dists = [
                    lp.local(cf.optim_var_at(j)).norm()
                    for j, lp in enumerate(lin_points[i])
                ]
                do_lin = np.max(lp_dists) > lp_dist_thresh

            do_lins.append(do_lin)

            if do_lin:
                potential_eta, potential_lam = compute_factor(cf, lie=lie)

                potentials_eta[i] = potential_eta
                potentials_lam[i] = potential_lam

                for j, var in enumerate(cf.optim_vars):
                    lin_points[i][j].update(var.data)

        # print(f"Linearised {np.sum(do_lins)} out of {len(do_lins)} factors.")
        return potentials_eta, potentials_lam, lin_points

    # loop for the iterative optimizer
    def _optimize_loop(
        self,
        start_iter: int,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        truncated_grad_loop: bool,
        relin_threshold: float = 0.1,
        damping: float = 0.0,
        dropout: float = 0.0,
        lp_dist_thresh: float = 0.1,
        **kwargs,
    ):
        # initialise messages with zeros
        vtof_msgs_eta = torch.zeros(
            self.n_edges, self.max_dofs, dtype=self.objective.dtype
        )
        vtof_msgs_lam = torch.zeros(
            self.n_edges, self.max_dofs, self.max_dofs, dtype=self.objective.dtype
        )
        ftov_msgs_eta = vtof_msgs_eta.clone()
        ftov_msgs_lam = vtof_msgs_lam.clone()

        # compute factor potentials for the first time
        potentials_eta = [None] * self.objective.size_cost_functions()
        potentials_lam = [None] * self.objective.size_cost_functions()
        lin_points = [
            [var.copy(new_name=f"{cf.name}_{var.name}_lp") for var in cf.optim_vars]
            for cf in self.cf_ordering
        ]
        potentials_eta, potentials_lam, lin_points = self._linearize(
            potentials_eta, potentials_lam, lin_points, lp_dist_thresh=None
        )

        converged_indices = torch.zeros_like(info.last_err).bool()
        for it_ in range(start_iter, start_iter + num_iter):

            potentials_eta, potentials_lam, lin_points = self._linearize(
                potentials_eta, potentials_lam, lin_points, lp_dist_thresh=None
            )

            msgs_eta, msgs_lam = pass_fac_to_var_messages(
                potentials_eta,
                potentials_lam,
                vtof_msgs_eta,
                vtof_msgs_lam,
                self.adj_var_dofs_nested,
            )

            # damping
            # damping = self.gbp_settings.get_damping(iters_since_relin)
            damping_arr = torch.full([len(msgs_eta)], damping)

            # dropout can be implemented through damping
            if dropout != 0.0:
                dropout_ixs = torch.rand(len(msgs_eta)) < dropout
                damping_arr[dropout_ixs] = 1.0

            ftov_msgs_eta = (1 - damping_arr[:, None]) * msgs_eta + damping_arr[
                :, None
            ] * ftov_msgs_eta
            ftov_msgs_lam = (1 - damping_arr[:, None, None]) * msgs_lam + damping_arr[
                :, None, None
            ] * ftov_msgs_lam

            (
                vtof_msgs_eta,
                vtof_msgs_lam,
                belief_eta,
                belief_lam,
            ) = pass_var_to_fac_messages(
                ftov_msgs_eta,
                ftov_msgs_lam,
                self.var_ix_for_edges,
                len(self.ordering._var_order),
                self.max_dofs,
            )

            # update beliefs
            belief_cov = torch.inverse(belief_lam)
            belief_mean = torch.matmul(belief_cov, belief_eta.unsqueeze(-1)).squeeze()
            for i, var in enumerate(self.ordering):
                var.update(data=belief_mean[i][None, :])

            # check for convergence
            with torch.no_grad():
                err = self.objective.error_squared_norm() / 2
                self._update_info(info, it_, err, converged_indices)
                if verbose:
                    print(f"GBP. Iteration: {it_+1}. " f"Error: {err.mean().item()}")
                converged_indices = self._check_convergence(err, info.last_err)
                info.status[
                    converged_indices.cpu().numpy()
                ] = NonlinearOptimizerStatus.CONVERGED
                if converged_indices.all():
                    break  # nothing else will happen at this point
                info.last_err = err

        info.status[
            info.status == NonlinearOptimizerStatus.START
        ] = NonlinearOptimizerStatus.MAX_ITERATIONS
        return info

    # loop for the iterative optimizer
    def _optimize_loop_lie(
        self,
        start_iter: int,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        truncated_grad_loop: bool,
        relin_threshold: float = 0.1,
        damping: float = 0.0,
        dropout: float = 0.0,
        lp_dist_thresh: float = 0.1,
        **kwargs,
    ):
        # initialise messages with zeros
        vtof_msgs = []
        ftov_msgs = []
        for cf in self.cf_ordering:
            for var in cf.optim_vars:
                vtof_msg_mu = var.copy(new_name=f"msg_{var.name}_to_{cf.name}")
                # mean of initial message doesn't matter as long as precision is zero
                vtof_msg_mu.data[:] = 0
                ftov_msg_mu = vtof_msg_mu.copy(new_name=f"msg_{cf.name}_to_{var.name}")
                vtof_msgs.append(Message(vtof_msg_mu))
                ftov_msgs.append(Message(ftov_msg_mu))

        # compute factor potentials for the first time
        potentials_eta = [None] * self.objective.size_cost_functions()
        potentials_lam = [None] * self.objective.size_cost_functions()
        lin_points = [
            [var.copy(new_name=f"{cf.name}_{var.name}_lp") for var in cf.optim_vars]
            for cf in self.cf_ordering
        ]
        potentials_eta, potentials_lam, lin_points = self._linearize(
            potentials_eta, potentials_lam, lin_points, lp_dist_thresh=None, lie=True
        )

        converged_indices = torch.zeros_like(info.last_err).bool()
        for it_ in range(start_iter, start_iter + num_iter):

            potentials_eta, potentials_lam, lin_points = self._linearize(
                potentials_eta,
                potentials_lam,
                lin_points,
                lp_dist_thresh=None,
                lie=True,
            )

            # damping
            # damping = self.gbp_settings.get_damping(iters_since_relin)
            damping_arr = torch.full([self.n_edges], damping)

            # dropout can be implemented through damping
            if dropout != 0.0:
                dropout_ixs = torch.rand(self.n_edges) < dropout
                damping_arr[dropout_ixs] = 1.0

            pass_fac_to_var_messages_lie(
                potentials_eta,
                potentials_lam,
                lin_points,
                vtof_msgs,
                ftov_msgs,
                self.adj_var_dofs_nested,
                damping_arr,
            )

            pass_var_to_fac_messages_and_update_beliefs_lie(
                ftov_msgs,
                vtof_msgs,
                self.ordering,
                self.var_ix_for_edges,
            )

            # check for convergence
            if it_ > 0:
                with torch.no_grad():
                    err = self.objective.error_squared_norm() / 2
                    self._update_info(info, it_, err, converged_indices)
                    if verbose:
                        print(
                            f"GBP. Iteration: {it_+1}. " f"Error: {err.mean().item()}"
                        )
                    converged_indices = self._check_convergence(err, info.last_err)
                    info.status[
                        converged_indices.cpu().numpy()
                    ] = NonlinearOptimizerStatus.CONVERGED
                    if converged_indices.all() and it_ > 1:
                        break  # nothing else will happen at this point
                    info.last_err = err

        info.status[
            info.status == NonlinearOptimizerStatus.START
        ] = NonlinearOptimizerStatus.MAX_ITERATIONS
        return info

    # `track_best_solution` keeps a **detached** copy (as in no gradient info)
    # of the best variables found, but it is optional to avoid unnecessary copying
    # if this is not needed
    def _optimize_impl(
        self,
        track_best_solution: bool = False,
        track_err_history: bool = False,
        verbose: bool = False,
        backward_mode: BackwardMode = BackwardMode.FULL,
        damping: float = 0.0,
        dropout: float = 0.0,
        **kwargs,
    ) -> NonlinearOptimizerInfo:
        if damping > 1.0 or damping < 0.0:
            raise NotImplementedError("Damping must be in between 0 and 1.")
        if dropout > 1.0 or dropout < 0.0:
            raise NotImplementedError("Dropout probability must be in between 0 and 1.")

        with torch.no_grad():
            info = self._init_info(track_best_solution, track_err_history, verbose)

        if verbose:
            print(
                f"GBP optimizer. Iteration: 0. " f"Error: {info.last_err.mean().item()}"
            )

        grad = False
        if backward_mode == BackwardMode.FULL:
            grad = True

        with torch.set_grad_enabled(grad):

            # if self.lie_groups:
            info = self._optimize_loop_lie(
                start_iter=0,
                num_iter=self.params.max_iterations,
                info=info,
                verbose=verbose,
                truncated_grad_loop=False,
                damping=damping,
                dropout=dropout,
                **kwargs,
            )
            # else:
            #     info = self._optimize_loop(
            #         start_iter=0,
            #         num_iter=self.params.max_iterations,
            #         info=info,
            #         verbose=verbose,
            #         truncated_grad_loop=False,
            #         damping=damping,
            #         dropout=dropout,
            #         **kwargs,
            #     )
            # If didn't coverge, remove misleading converged_iter value
            info.converged_iter[
                info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
            ] = -1
            return info
