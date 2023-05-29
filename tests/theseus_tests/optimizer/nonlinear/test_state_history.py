# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401

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
