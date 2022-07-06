# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from itertools import count
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import theseus as th
import theseus.constants
from theseus.core import CostFunction, Objective
from theseus.geometry import Manifold
from theseus.optimizer import Optimizer, VariableOrdering
from theseus.optimizer.nonlinear.nonlinear_optimizer import (
    BackwardMode,
    NonlinearOptimizerInfo,
    NonlinearOptimizerStatus,
)

"""
TODO
 - solving inverse problem to compute message mean
"""


"""
Utitily functions
"""


# Same of NonlinearOptimizerParams but without step size
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


class GBPSchedule(Enum):
    SYNCHRONOUS = 0
    RANDOM = 1


def synchronous_schedule(max_iters, n_edges) -> torch.Tensor:
    return torch.full([max_iters, n_edges], True)


def random_schedule(max_iters, n_edges) -> torch.Tensor:
    schedule = torch.full([max_iters, n_edges], False)
    ixs = torch.randint(0, n_edges, [max_iters])
    schedule[torch.arange(max_iters), ixs] = True
    return schedule


# Initialises message precision to zero
class Message(th.ManifoldGaussian):
    def __init__(
        self,
        mean: Sequence[Manifold],
        precision: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ):
        if precision is None:
            dof = sum([v.dof() for v in mean])
            precision = torch.zeros(mean[0].shape[0], dof, dof).to(
                dtype=mean[0].dtype, device=mean[0].device
            )
        super(Message, self).__init__(mean, precision=precision, name=name)

    # sets mean to the group identity and zero precision matrix
    def zero_message(self):
        new_mean = []
        batch_size = self.mean[0].shape[0]
        for var in self.mean:
            if var.__class__ == th.Vector:
                new_mean_i = var.__class__(var.dof())
            else:
                new_mean_i = var.__class__()
            repeats = torch.ones(var.ndim, dtype=int)
            repeats[0] = batch_size
            new_mean_i = new_mean_i.data.repeat(repeats.tolist())
            new_mean_i = new_mean_i.to(dtype=self.dtype, device=self.device)
            new_mean.append(new_mean_i)
        new_precision = torch.zeros(batch_size, self.dof, self.dof).to(
            dtype=self.dtype, device=self.device
        )
        self.update(mean=new_mean, precision=new_precision)


"""
GBP functions
"""


class Factor:
    _ids = count(0)

    def __init__(
        self,
        cf: CostFunction,
        name: Optional[str] = None,
        lin_system_damping: float = 1e-6,
    ):
        self._id = next(Factor._ids)
        if name:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}__{self._id}"

        self.cf = cf
        self.lin_system_damping = lin_system_damping

        # batch_size of the vectorized factor. In general != objective.batch_size.
        # They are equal without vectorization or for unique cost function schema.
        self.batch_size = cf.optim_var_at(0).shape[0]

        device = cf.optim_var_at(0).device
        dtype = cf.optim_var_at(0).dtype
        self._dof = sum([var.dof() for var in cf.optim_vars])
        self.potential_eta = torch.zeros(self.batch_size, self.dof).to(
            dtype=dtype, device=device
        )
        self.potential_lam = torch.zeros(self.batch_size, self.dof, self.dof).to(
            dtype=dtype, device=device
        )
        self.lin_point = [
            var.copy(new_name=f"{cf.name}_{var.name}_lp") for var in cf.optim_vars
        ]

        self.steps_since_lin = torch.zeros(
            self.batch_size, device=device, dtype=torch.int
        )

    # Linearizes factors at current belief if beliefs have deviated
    # from the linearization point by more than the threshold.
    def linearize(
        self,
        relin_threshold: float = None,
        lie=True,
    ):
        self.steps_since_lin += 1

        if relin_threshold is None:
            do_lin = torch.full(
                [self.batch_size],
                True,
                device=self.cf.optim_var_at(0).device,
            )
        else:
            lp_dists = torch.cat(
                [
                    lp.local(self.cf.optim_var_at(j)).norm(dim=1)[..., None]
                    for j, lp in enumerate(self.lin_point)
                ],
                dim=1,
            )
            max_dists = lp_dists.max(dim=1)[0]
            do_lin = max_dists > relin_threshold

        if torch.sum(do_lin) > 0:  # if any factor in the batch needs relinearization
            J, error = self.cf.weighted_jacobians_error()

            J_stk = torch.cat(J, dim=-1)

            lam = torch.bmm(J_stk.transpose(-2, -1), J_stk)
            eta = -torch.matmul(J_stk.transpose(-2, -1), error.unsqueeze(-1))
            if lie is False:
                optim_vars_stk = torch.cat([v.data for v in self.cf.optim_vars], dim=-1)
                eta = eta + torch.matmul(lam, optim_vars_stk.unsqueeze(-1))
            eta = eta.squeeze(-1)

            self.potential_eta[do_lin] = eta[do_lin]
            self.potential_lam[do_lin] = lam[do_lin]

            for j, var in enumerate(self.cf.optim_vars):
                self.lin_point[j].update(var.data, batch_ignore_mask=~do_lin)

            self.steps_since_lin[do_lin] = 0

    # Compute all outgoing messages from the factor.
    def comp_mess(
        self,
        vtof_msgs,
        ftov_msgs,
        damping,
    ):
        num_optim_vars = self.cf.num_optim_vars()
        new_messages = []

        sdim = 0
        for v in range(num_optim_vars):
            eta_factor = self.potential_eta.clone()
            lam_factor = self.potential_lam.clone()
            lam_factor_copy = lam_factor.clone()

            # Take product of factor with incoming messages.
            # Convert mesages to tangent space at linearisation point.
            start = 0
            for i in range(num_optim_vars):
                var_dofs = self.cf.optim_var_at(i).dof()
                if i != v:
                    eta_mess, lam_mess = th.local_gaussian(
                        self.lin_point[i], vtof_msgs[i], return_mean=False
                    )
                    eta_factor[:, start : start + var_dofs] += eta_mess
                    lam_factor[
                        :, start : start + var_dofs, start : start + var_dofs
                    ] += lam_mess

                start += var_dofs

            dofs = self.cf.optim_var_at(v).dof()

            if torch.allclose(lam_factor, lam_factor_copy) and num_optim_vars > 1:
                # print(self.cf.name, "---> not updating, incoming precision is zero")
                new_mess = Message([self.cf.optim_var_at(v).copy()])
                new_mess.zero_message()

            else:
                # print(self.cf.name, "---> sending message")
                # Divide up parameters of distribution
                eo = eta_factor[:, sdim : sdim + dofs]
                eno = torch.cat(
                    (eta_factor[:, :sdim], eta_factor[:, sdim + dofs :]), dim=1
                )

                loo = lam_factor[:, sdim : sdim + dofs, sdim : sdim + dofs]
                lono = torch.cat(
                    (
                        lam_factor[:, sdim : sdim + dofs, :sdim],
                        lam_factor[:, sdim : sdim + dofs, sdim + dofs :],
                    ),
                    dim=2,
                )
                lnoo = torch.cat(
                    (
                        lam_factor[:, :sdim, sdim : sdim + dofs],
                        lam_factor[:, sdim + dofs :, sdim : sdim + dofs],
                    ),
                    dim=1,
                )
                lnono = torch.cat(
                    (
                        torch.cat(
                            (
                                lam_factor[:, :sdim, :sdim],
                                lam_factor[:, :sdim, sdim + dofs :],
                            ),
                            dim=2,
                        ),
                        torch.cat(
                            (
                                lam_factor[:, sdim + dofs :, :sdim],
                                lam_factor[:, sdim + dofs :, sdim + dofs :],
                            ),
                            dim=2,
                        ),
                    ),
                    dim=1,
                )

                new_mess_lam = loo - lono @ torch.linalg.inv(lnono) @ lnoo
                new_mess_eta = eo - torch.bmm(
                    torch.bmm(lono, torch.linalg.inv(lnono)), eno.unsqueeze(-1)
                ).squeeze(-1)

                # damping in tangent space at linearisation point as message
                # is already in this tangent space. Could equally do damping
                # in the tangent space of the new or old message mean.
                # mean damping
                do_damping = torch.logical_and(damping[v] > 0, self.steps_since_lin > 0)
                if do_damping.sum() > 0:
                    damping_check = torch.logical_and(
                        new_mess_lam.count_nonzero(1, 2) != 0,
                        ftov_msgs[v].precision.count_nonzero(1, 2) != 0,
                    )
                    do_damping = torch.logical_and(do_damping, damping_check)
                    if do_damping.sum() > 0:
                        prev_mess_mean, prev_mess_lam = th.local_gaussian(
                            self.lin_point[v], ftov_msgs[v], return_mean=True
                        )
                        new_mess_mean = torch.bmm(
                            torch.inverse(new_mess_lam), new_mess_eta.unsqueeze(-1)
                        ).squeeze(-1)
                        damping[v][~do_damping] = 0.0
                        new_mess_mean = (
                            1 - damping[v][:, None]
                        ) * new_mess_mean + damping[v][:, None] * prev_mess_mean
                        new_mess_eta = torch.bmm(
                            new_mess_lam, new_mess_mean.unsqueeze(-1)
                        ).squeeze(-1)

                new_mess_lam = th.DenseSolver._apply_damping(
                    new_mess_lam,
                    self.lin_system_damping,
                    ellipsoidal=True,
                    eps=1e-8,
                )

                new_mess_mean = th.LUDenseSolver._solve_sytem(
                    new_mess_eta[..., None], new_mess_lam
                )

                new_mess = th.retract_gaussian(
                    self.lin_point[v], new_mess_mean, new_mess_lam
                )

            new_messages.append(new_mess)

            sdim += dofs

        # update messages
        for v in range(num_optim_vars):
            ftov_msgs[v].update(
                mean=new_messages[v].mean, precision=new_messages[v].precision
            )

        return new_messages

    @property
    def dof(self) -> int:
        return self._dof


# Follows notation from https://arxiv.org/pdf/2202.03314.pdf
class GaussianBeliefPropagation(Optimizer, abc.ABC):
    def __init__(
        self,
        objective: Objective,
        vectorize: bool = True,
        abs_err_tolerance: float = 1e-10,
        rel_err_tolerance: float = 1e-8,
        max_iterations: int = 20,
    ):
        super().__init__(objective, vectorize=vectorize)

        # ordering is required to identify which messages to send where
        self.ordering = VariableOrdering(objective, default_order=True)

        self.params = GBPOptimizerParams(
            abs_err_tolerance, rel_err_tolerance, max_iterations
        )

        # create array for indexing the messages
        var_ixs_nested = [
            [self.ordering.index_of(var.name) for var in cf.optim_vars]
            for cf in self.objective.cost_functions.values()
        ]
        var_ixs = [item for sublist in var_ixs_nested for item in sublist]
        self.var_ix_for_edges = torch.tensor(var_ixs).long()

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
    GBP functions
    """

    def _pass_var_to_fac_messages(
        self,
        update_belief=True,
    ):
        for i, var in enumerate(self.ordering):

            # Collect all incoming messages in the tangent space at the current belief
            taus = []  # message means
            lams_tp = []  # message lams
            for j, msg in enumerate(self.ftov_msgs):
                if self.var_ix_for_edges[j] == i:
                    # print(msg.mean, msg.precision)
                    tau, lam_tp = th.local_gaussian(var, msg, return_mean=True)
                    taus.append(tau[None, ...])
                    lams_tp.append(lam_tp[None, ...])

            taus = torch.cat(taus)
            lams_tp = torch.cat(lams_tp)

            lam_tau = lams_tp.sum(dim=0)

            # Compute outgoing messages
            ix = 0
            for j, msg in enumerate(self.ftov_msgs):
                if self.var_ix_for_edges[j] == i:
                    taus_inc = torch.cat((taus[:ix], taus[ix + 1 :]))
                    lams_inc = torch.cat((lams_tp[:ix], lams_tp[ix + 1 :]))

                    lam_a = lams_inc.sum(dim=0)
                    if lam_a.count_nonzero() == 0:
                        self.vtof_msgs[j].zero_message()
                    else:
                        inv_lam_a = torch.linalg.inv(lam_a)
                        sum_taus = torch.matmul(lams_inc, taus_inc.unsqueeze(-1)).sum(
                            dim=0
                        )
                        tau_a = torch.matmul(inv_lam_a, sum_taus).squeeze(-1)
                        new_mess = th.retract_gaussian(var, tau_a, lam_a)
                        self.vtof_msgs[j].update(new_mess.mean, new_mess.precision)
                    ix += 1

            # update belief mean and variance
            # if no incoming messages then leave current belief unchanged
            if update_belief and lam_tau.count_nonzero() != 0:
                inv_lam_tau = torch.inverse(lam_tau)
                sum_taus = torch.matmul(lams_tp, taus.unsqueeze(-1)).sum(dim=0)
                tau = torch.matmul(inv_lam_tau, sum_taus).squeeze(-1)

                new_belief = th.retract_gaussian(var, tau, lam_tau)
                self.beliefs[i].update(new_belief.mean, new_belief.precision)

    def _linearize_factors(self, relin_threshold: float = None):
        relins = 0
        for factor in self.factors:
            factor.linearize(relin_threshold=relin_threshold)
            relins += int((factor.steps_since_lin == 0).sum().item())

        return relins

    def _pass_fac_to_var_messages(
        self,
        schedule: torch.Tensor,
        damping: torch.Tensor,
    ):

        # USE THE SCHEDULE!!!!!

        start = 0
        start_d = 0
        for j, factor in enumerate(self.factors):
            num_optim_vars = factor.cf.num_optim_vars()
            n_factors = num_optim_vars * factor.batch_size
            damping_tsr = damping[start_d : start_d + n_factors].reshape(
                num_optim_vars, factor.batch_size
            )
            start_d += n_factors

            if self.objective.vectorized:
                # prepare vectorized messages
                ixs = torch.tensor(self.objective.vectorized_msg_ixs[j])
                vtof_msgs: List[Message] = []
                ftov_msgs: List[Message] = []
                for var in factor.cf.optim_vars:
                    mean_vtof_msgs = var.copy()
                    mean_ftov_msgs = var.copy()
                    mean_data_vtof_msgs = torch.cat(
                        [self.vtof_msgs[i].mean[0].data for i in ixs]
                    )
                    mean_data_ftov_msgs = torch.cat(
                        [self.ftov_msgs[i].mean[0].data for i in ixs]
                    )
                    mean_vtof_msgs.update(data=mean_data_vtof_msgs)
                    mean_ftov_msgs.update(data=mean_data_ftov_msgs)
                    precision_vtof_msgs = torch.cat(
                        [self.vtof_msgs[i].precision for i in ixs]
                    )
                    precision_ftov_msgs = torch.cat(
                        [self.ftov_msgs[i].precision for i in ixs]
                    )

                    vtof_msg = Message(
                        mean=[mean_vtof_msgs], precision=precision_vtof_msgs
                    )
                    ftov_msg = Message(
                        mean=[mean_ftov_msgs], precision=precision_ftov_msgs
                    )
                    vtof_msgs.append(vtof_msg)
                    ftov_msgs.append(ftov_msg)

                    ixs += 1
            else:
                vtof_msgs = self.vtof_msgs[start : start + num_optim_vars]
                ftov_msgs = self.ftov_msgs[start : start + num_optim_vars]

            factor.comp_mess(vtof_msgs, ftov_msgs, damping_tsr)

            if self.objective.vectorized:
                # fill in messages using vectorized messages
                ixs = torch.tensor(self.objective.vectorized_msg_ixs[j])
                for ftov_msg in ftov_msgs:
                    start_idx = 0
                    for ix in ixs:
                        v_slice = slice(
                            start_idx, start_idx + self.objective.batch_size
                        )
                        self.ftov_msgs[ix].update(
                            mean=[ftov_msg.mean[0][v_slice]],
                            precision=ftov_msg.precision[v_slice],
                        )
                        start_idx += self.objective.batch_size
                    ixs += 1

            start += num_optim_vars

    """
    Optimization loop functions
    """

    # loop for the iterative optimizer
    def _optimize_loop(
        self,
        start_iter: int,
        num_iter: int,
        info: NonlinearOptimizerInfo,
        verbose: bool,
        truncated_grad_loop: bool,
        relin_threshold: float,
        damping: float,
        dropout: float,
        schedule: GBPSchedule,
        lin_system_damping: float,
        clear_messages: bool = True,
        **kwargs,
    ):
        if damping > 1.0 or damping < 0.0:
            raise ValueError(f"Damping must be between 0 and 1. Got {damping}.")
        if dropout > 1.0 or dropout < 0.0:
            raise ValueError(
                f"Dropout probability must be between 0 and 1. Got {dropout}."
            )

        if clear_messages:
            # initialise messages with zeros
            self.vtof_msgs: List[Message] = []
            self.ftov_msgs: List[Message] = []
            for cf in self.objective.cost_functions.values():
                for var in cf.optim_vars:
                    # Set mean of initial message to identity of the group
                    # doesn't matter what it is as long as precision is zero
                    vtof_msg = Message(
                        [var.copy()], name=f"msg_{var.name}_to_{cf.name}"
                    )
                    ftov_msg = Message(
                        [var.copy()], name=f"msg_{cf.name}_to_{var.name}"
                    )
                    vtof_msg.zero_message()
                    ftov_msg.zero_message()
                    self.vtof_msgs.append(vtof_msg)
                    self.ftov_msgs.append(ftov_msg)

        # initialise ManifoldGaussian for belief
        self.beliefs: List[th.ManifoldGaussian] = []
        for var in self.ordering:
            self.beliefs.append(th.ManifoldGaussian([var]))

        self.n_individual_factors = (
            len(self.objective.cost_functions) * self.objective.batch_size
        )
        if self.objective.vectorized:
            cf_iterator = iter(self.objective.vectorized_cost_fns)
        else:
            cf_iterator = self.objective._get_iterator()

        # compute factor potentials for the first time
        self.factors: List[Factor] = []
        for cost_function in cf_iterator:
            self.factors.append(
                Factor(
                    cost_function,
                    name=cost_function.name,
                    lin_system_damping=lin_system_damping,
                )
            )
        relins = self._linearize_factors()

        self.n_edges = sum(
            [factor.cf.num_optim_vars() * factor.batch_size for factor in self.factors]
        )
        if schedule == GBPSchedule.RANDOM:
            ftov_schedule = random_schedule(self.params.max_iterations, self.n_edges)
        elif schedule == GBPSchedule.SYNCHRONOUS:
            ftov_schedule = synchronous_schedule(
                self.params.max_iterations, self.n_edges
            )

        self.belief_history = {}
        self.ftov_msgs_history = {}

        converged_indices = torch.zeros_like(info.last_err).bool()
        for it_ in range(start_iter, start_iter + num_iter):
            self.ftov_msgs_history[it_] = [msg.copy() for msg in self.ftov_msgs]
            self.belief_history[it_] = [belief.copy() for belief in self.beliefs]

            # damping
            damping_arr = torch.full(
                [self.n_edges],
                damping,
                device=self.ordering[0].device,
                dtype=self.ordering[0].dtype,
            )
            # dropout can be implemented through damping
            if dropout != 0.0:
                dropout_ixs = torch.rand(self.n_edges) < dropout
                damping_arr[dropout_ixs] = 1.0

            t0 = time.time()
            relins = self._linearize_factors(relin_threshold)
            t_relin = time.time() - t0

            t1 = time.time()
            self._pass_fac_to_var_messages(ftov_schedule[it_], damping_arr)
            t_ftov = time.time() - t1

            t1 = time.time()
            self._pass_var_to_fac_messages(update_belief=True)
            t_vtof = time.time() - t1

            t_vec = 0.0
            if self.objective.vectorized:
                t1 = time.time()
                self.objective.update_vectorization_if_needed(compute_caches=False)
                t_vec = time.time() - t1

            t_tot = time.time() - t0
            print(
                f"Timings ----- relin {t_relin:.4f}, ftov {t_ftov:.4f}, vtof {t_vtof:.4f},"
                f" vectorization {t_vec:.4f}, TOTAL {t_tot:.4f}"
            )

            # check for convergence
            if it_ >= 0:
                with torch.no_grad():
                    err = self.objective.error_squared_norm() / 2
                    self._update_info(info, it_, err, converged_indices)
                    if verbose:
                        print(
                            f"GBP. Iteration: {it_+1}. Error: {err.mean().item():.4f}. "
                            f"Relins: {relins} / {self.n_individual_factors}"
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
        relin_threshold: float = 0.1,
        damping: float = 0.0,
        dropout: float = 0.0,
        schedule: GBPSchedule = GBPSchedule.SYNCHRONOUS,
        lin_system_damping: float = 1e-6,
        **kwargs,
    ) -> NonlinearOptimizerInfo:
        with torch.no_grad():
            info = self._init_info(track_best_solution, track_err_history, verbose)

        if verbose:
            print(
                f"GBP optimizer. Iteration: 0. " f"Error: {info.last_err.mean().item()}"
            )

        if backward_mode == BackwardMode.FULL:
            info = self._optimize_loop(
                start_iter=0,
                num_iter=self.params.max_iterations,
                info=info,
                verbose=verbose,
                truncated_grad_loop=False,
                relin_threshold=relin_threshold,
                damping=damping,
                dropout=dropout,
                schedule=schedule,
                lin_system_damping=lin_system_damping,
                **kwargs,
            )

            # If didn't coverge, remove misleading converged_iter value
            info.converged_iter[
                info.status == NonlinearOptimizerStatus.MAX_ITERATIONS
            ] = -1
            return info

        elif backward_mode in [BackwardMode.IMPLICIT, BackwardMode.TRUNCATED]:
            if backward_mode == BackwardMode.IMPLICIT:
                backward_num_iterations = 1
            else:
                if "backward_num_iterations" not in kwargs:
                    raise ValueError(
                        "backward_num_iterations expected but not received"
                    )
                if kwargs["backward_num_iterations"] > self.params.max_iterations:
                    warnings.warn(
                        f"Input backward_num_iterations "
                        f"(={kwargs['backward_num_iterations']}) > "
                        f"max_iterations (={self.params.max_iterations}). "
                        f"Using backward_num_iterations=max_iterations."
                    )
                backward_num_iterations = min(
                    kwargs["backward_num_iterations"], self.params.max_iterations
                )

            num_no_grad_iter = self.params.max_iterations - backward_num_iterations
            with torch.no_grad():
                self._optimize_loop(
                    start_iter=0,
                    num_iter=num_no_grad_iter,
                    info=info,
                    verbose=verbose,
                    truncated_grad_loop=False,
                    relin_threshold=relin_threshold,
                    damping=damping,
                    dropout=dropout,
                    schedule=schedule,
                    lin_system_damping=lin_system_damping,
                    **kwargs,
                )

            grad_loop_info = self._init_info(
                track_best_solution, track_err_history, verbose
            )
            self._optimize_loop(
                start_iter=0,
                num_iter=backward_num_iterations,
                info=grad_loop_info,
                verbose=verbose,
                truncated_grad_loop=True,
                relin_threshold=relin_threshold,
                damping=damping,
                dropout=dropout,
                schedule=schedule,
                lin_system_damping=lin_system_damping,
                clear_messages=False,
                **kwargs,
            )

            # Adds grad_loop_info results to original info
            self._merge_infos(
                grad_loop_info, num_no_grad_iter, backward_num_iterations, info
            )

            return info
        else:
            raise ValueError("Unrecognized backward mode")
