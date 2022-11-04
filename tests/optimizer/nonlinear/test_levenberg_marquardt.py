# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th

from theseus.constants import __FROM_THESEUS_LAYER_TOKEN__
from tests.optimizer.nonlinear.common import run_nonlinear_least_squares_check


@pytest.fixture
def mock_objective():
    objective = th.Objective()
    v1 = th.Vector(1, name="v1")
    v2 = th.Vector(1, name="v2")
    objective.add(th.Difference(v1, v2, th.ScaleCostWeight(1.0)))
    return objective


@pytest.mark.parametrize("damping", [0.0, 0.001, 0.01, 0.1])
@pytest.mark.parametrize("ellipsoidal_damping", [True, False])
@pytest.mark.parametrize("adaptive_damping", [True, False])
def test_levenberg_marquardt(damping, ellipsoidal_damping, adaptive_damping):
    run_nonlinear_least_squares_check(
        th.LevenbergMarquardt,
        {
            "damping": damping,
            "ellipsoidal_damping": ellipsoidal_damping,
            "adaptive_damping": adaptive_damping,
            "damping_eps": 0.0,
            __FROM_THESEUS_LAYER_TOKEN__: True,
        },
        singular_check=damping < 0.001,
    )


def test_ellipsoidal_damping_compatibility(mock_objective):
    mock_objective.update({"v1": torch.ones(1, 1), "v2": torch.zeros(1, 1)})
    for lsc in [th.LUDenseSolver, th.CholeskyDenseSolver]:
        optimizer = th.LevenbergMarquardt(mock_objective, lsc)
        optimizer.optimize(
            **{"ellipsoidal_damping": True, __FROM_THESEUS_LAYER_TOKEN__: True}
        )
        optimizer.optimize(**{"damping_eps": 0.1, __FROM_THESEUS_LAYER_TOKEN__: True})

    for lsc in [th.CholmodSparseSolver]:
        optimizer = th.LevenbergMarquardt(mock_objective, lsc)
        with pytest.raises(RuntimeError):
            optimizer.optimize(
                **{"ellipsoidal_damping": True, __FROM_THESEUS_LAYER_TOKEN__: True}
            )
        with pytest.raises(RuntimeError):
            optimizer.optimize(
                **{"damping_eps": 0.1, __FROM_THESEUS_LAYER_TOKEN__: True}
            )


@pytest.mark.cudaext
def test_ellipsoidal_damping_compatibility_cuda(mock_objective):
    if not torch.cuda.is_available():
        return
    mock_objective.to(device="cuda", dtype=torch.double)
    batch_size = 2
    mock_objective.update(
        {
            "v1": torch.ones(batch_size, 1, device="cuda", dtype=torch.double),
            "v2": torch.zeros(batch_size, 1, device="cuda", dtype=torch.double),
        }
    )
    for lsc in [th.LUCudaSparseSolver]:
        optimizer = th.LevenbergMarquardt(
            mock_objective, lsc, linear_solver_kwargs={"batch_size": batch_size}
        )
        optimizer.optimize(
            **{"ellipsoidal_damping": True, __FROM_THESEUS_LAYER_TOKEN__: True}
        )
        optimizer.optimize(**{"damping_eps": 0.1, __FROM_THESEUS_LAYER_TOKEN__: True})
