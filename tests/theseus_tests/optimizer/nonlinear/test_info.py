# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th


@pytest.mark.parametrize(
    "var_type", [(th.Vector, 1), (th.Vector, 2), (th.SE2, None), (th.SE3, None)]
)
@pytest.mark.parametrize("batch_size", [1, 10])
def test_state_history(var_type, batch_size):
    cls_, dof = var_type

    rand_args = (batch_size, dof) if cls_ == th.Vector else (batch_size,)
    v1 = cls_(tensor=cls_.rand(*rand_args).tensor, name="v1")
    v2 = cls_(tensor=cls_.rand(*rand_args).tensor, name="v2")
    w = th.ScaleCostWeight(1.0)

    objective = th.Objective()
    objective.add(th.Difference(v1, v2, w))

    max_iters = 10
    optimizer = th.GaussNewton(objective, max_iterations=max_iters)
    layer = th.TheseusLayer(optimizer)

    _, info = layer.forward(optimizer_kwargs={"track_state_history": True})

    for var in objective.optim_vars.values():
        assert var.name in info.state_history
        assert info.state_history[var.name].shape == (objective.batch_size,) + tuple(
            var.shape[1:]
        ) + (max_iters + 1,)


@pytest.mark.parametrize("batch_size", [1, 10])
def test_track_best_solution_matrix_vars(batch_size):
    x = th.SO3(name="x")
    y = th.SO3(name="y")
    objective = th.Objective()
    objective.add(th.Difference(x, y, th.ScaleCostWeight(1.0), name="cf"))
    optim = th.LevenbergMarquardt(objective, vectorize=True)
    objective.update({"x": torch.randn(batch_size, 3, 3)})
    info = optim.optimize(track_best_solution=True, backward_mode="implicit")
    assert info.best_solution["x"].shape == (batch_size, 3, 3)
    # Call to optimize  with track_best_solution=True used to fail
    # when tracking matrix vars (e.g., SO3/SE3)
