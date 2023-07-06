# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from omegaconf import OmegaConf

import examples.pose_graph.pose_graph_synthetic as pgo

from tests.theseus_tests.decorators import run_if_baspacho


@pytest.fixture
def default_cfg():
    cfg = OmegaConf.load("examples/configs/pose_graph/pose_graph_synthetic.yaml")
    cfg.outer_optim.num_epochs = 1
    cfg.outer_optim.max_num_batches = 4
    cfg.batch_size = 16
    cfg.num_poses = 64
    cfg.profile = False
    cfg.savemat = False
    cfg.inner_optim.optimizer_kwargs.verbose = False
    return cfg


@pytest.mark.parametrize(
    "linear_solver_cls",
    ["CholeskyDenseSolver", "LUCudaSparseSolver", "CholmodSparseSolver"],
)
def test_pgo_losses(default_cfg, linear_solver_cls):
    # for everything except cholmod (need to turn off adaptive damping for that one)
    expected_losses = [
        -0.29886279606812166,
        -0.3054215856589109,
        -0.27485602196709225,
        -0.3005231105990632,
    ]

    default_cfg.inner_optim.linear_solver_cls = linear_solver_cls
    if linear_solver_cls == "LUCudaSparseSolver":
        if not torch.cuda.is_available():
            return
        default_cfg.device = "cuda:0"
    else:
        if linear_solver_cls == "CholmodSparseSolver":
            default_cfg.inner_optim.optimizer_kwargs.adaptive_damping = False
            expected_losses = [
                -0.2988627961673474,
                -0.30542158576120654,
                -0.27485602213117594,
                -0.3005231108739672,
            ]
        default_cfg.device = "cpu"
    losses = pgo.run(default_cfg)
    print(losses)

    for loss, expected_loss in zip(losses[0], expected_losses):
        assert loss == pytest.approx(expected_loss, rel=1e-10, abs=1e-10)


@run_if_baspacho()
def test_pgo_losses_baspacho(default_cfg):
    # for everything except cholmod (need to turn off adaptive damping for that one)
    expected_losses = [
        -0.2988627960682926,
        -0.30542158565900696,
        -0.27485602196705955,
        -0.3005231105991407,
    ]

    default_cfg.inner_optim.linear_solver_cls = "BaspachoSparseSolver"
    default_cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    losses = pgo.run(default_cfg)
    print(losses)

    for loss, expected_loss in zip(losses[0], expected_losses):
        assert loss == pytest.approx(expected_loss, rel=1e-10, abs=1e-10)
