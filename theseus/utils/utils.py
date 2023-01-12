# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Any, Callable, List, Optional, Type

import torch
import torch.nn as nn

import theseus as th


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int,
    act_name: str = "relu",
) -> torch.nn.Sequential:
    # assign types
    act: Type[nn.Module]
    mods: List[nn.Module]

    # find activation
    if act_name == "relu":
        act = nn.ReLU
    elif act_name == "elu":
        act = nn.ELU
    elif act_name == "tanh":
        act = nn.Tanh
    else:
        raise NotImplementedError()

    # construct sequential list of modules
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


# Inputs are a batched matrix and batched rows/cols for indexing
# Sizes are:
#      matrix -> batch_size x num_rows x num_cols
#      rows/cols -> batch_size x num_points
#
# If the matrix's batch size is 1, broadcasting is supported by expanding the matrix
# so that it has the batch size of rows and cols.
#
# Returns:
#      data -> batch_size x num_points
# where
#      data[i, j] = matrix[i, rows[i, j], cols[i, j]]
def gather_from_rows_cols(matrix: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor):
    if matrix.shape[0] == 1:
        matrix = matrix.expand(rows.shape[0], -1, -1)
    assert matrix.ndim == 3 and rows.ndim == 2 and rows.ndim == 2
    assert matrix.shape[0] == rows.shape[0] and matrix.shape[0] == cols.shape[0]
    assert rows.shape[1] == cols.shape[1]
    assert rows.min() >= 0 and rows.max() < matrix.shape[1]
    assert cols.min() >= 0 and cols.max() < matrix.shape[2]

    aux_idx = torch.arange(matrix.shape[0]).unsqueeze(-1).to(matrix.device)
    return matrix[aux_idx, rows, cols]


# Computes jacobians of a function of LieGroup arguments using finite differences
#   function: a callable representing the function whose jacobians will be computed
#   group_args: the group values at which the jacobians should be computed
#   function_dim: the dimension of the function. If None, it uses the dimension of the
#      group args.
#   delta_mag: magnitude of the differences used to compute the jacobian
def numeric_jacobian(
    function: Callable[[List[th.LieGroup]], th.LieGroup],
    group_args: List[th.LieGroup],
    function_dim: Optional[int] = None,
    delta_mag: float = 1e-3,
):
    batch_size = max([group_arg.shape[0] for group_arg in group_args])

    def _compute(group_idx):
        dof = group_args[group_idx].dof()
        function_dim_ = function_dim or dof
        jac = torch.zeros(
            batch_size, function_dim_, dof, dtype=group_args[group_idx].dtype
        )
        for d in range(dof):
            delta = torch.zeros(1, dof).to(
                device=group_args[0].device, dtype=group_args[group_idx].dtype
            )
            delta[:, d] = delta_mag

            group_plus = group_args[group_idx].retract(delta)
            group_minus = group_args[group_idx].retract(-delta)
            group_plus_args = [g for g in group_args]
            group_plus_args[group_idx] = group_plus
            group_minus_args = [g for g in group_args]
            group_minus_args[group_idx] = group_minus

            diff = function(group_minus_args).local(function(group_plus_args))
            jac[:, :, d] = diff / (2 * delta_mag)
        return jac

    jacs = []
    for group_idx in range(len(group_args)):
        jacs.append(_compute(group_idx))
    return jacs


# A basic timer utility that adapts to the device. Useful for removing
# boilerplate code when benchmarking tasks.
# For CPU it uses time.perf_counter_ns()
# For GPU it uses torch.cuda.Event()
#
# Usage:
#
# from thesus.utils import Timer
#
# with Timer("cuda:0") as timer:
#    do_some_stuff()
# print(timer.elapsed_time)
class Timer:
    def __init__(self, device: th.DeviceType) -> None:
        self.device = torch.device(device)
        self.elapsed_time = 0.0

    def __enter__(self) -> "Timer":
        if self.device.type == "cuda":
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_time = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.device.type == "cuda":
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self._start_event.elapsed_time(self._end_event) / 1e3
        else:
            self.elapsed_time = (time.perf_counter_ns() - self._start_time) / 1e9
