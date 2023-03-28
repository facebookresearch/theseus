# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import theseus as th
import torch

from theseus.utils import check_jacobians


def test_nonholonomic():
    rng = torch.Generator()
    rng.manual_seed(0)
    # Check SE2 pose version
    pose = th.SE2.rand(10, generator=rng)
    vel = th.Vector.rand(10, 3, generator=rng)
    w = th.ScaleCostWeight(1.0)
    cf = th.eb.Nonholonomic(pose, vel, w)
    check_jacobians(cf, num_checks=100, tol=1e-5)
    # Check Vector3 pose version
    pose = th.Vector.rand(10, 3, generator=rng)
    cf = th.eb.Nonholonomic(pose, vel, w)
    check_jacobians(cf, num_checks=100, tol=1e-5)


def test_hinge_cost():
    rng = torch.Generator()
    rng.manual_seed(0)
    batch_size = 10
    how_many = 4

    def _rand_chunk():
        return torch.rand(batch_size, how_many, generator=rng)

    for limit in [0.0, 1.0]:
        threshold = 0.0 if limit == 0.0 else 0.2
        vector = torch.zeros(batch_size, 3 * how_many)

        # Vector is created so that [below_th, within_th, above_th]
        vector[:, :how_many] = -_rand_chunk() - limit
        if limit == 0.0:
            vector[:, how_many : 2 * how_many] = 0.0
        else:
            vector[:, how_many : 2 * how_many] = (
                -limit + threshold + 0.1 * _rand_chunk()
            )
        vector[:, 2 * how_many :] = limit + _rand_chunk()
        v = th.Vector(tensor=vector)
        cf = th.eb.HingeCost(v, -limit, limit, threshold, th.ScaleCostWeight(1.0))

        jacobians, error = cf.jacobians()
        assert jacobians[0].shape == (batch_size, 3 * how_many, 3 * how_many)
        assert error.shape == (batch_size, 3 * how_many)
        # Middle section error must be == 0
        assert (error[:, how_many : 2 * how_many] == 0).all().item()
        # Check jacobians
        # All jacobians are equal and the number of nz elements is how_many times -1 and
        # and how_many 1, for each batch index
        nn_zero = batch_size * how_many
        assert jacobians[0].abs().sum() == 2 * nn_zero
        assert jacobians[0][:, :how_many, :how_many].sum() == -nn_zero
        assert jacobians[0][:, 2 * how_many :, 2 * how_many :].sum() == nn_zero

        # Throw in some random checks as well, why not
        check_jacobians(cf, num_checks=100, tol=1e-5)
