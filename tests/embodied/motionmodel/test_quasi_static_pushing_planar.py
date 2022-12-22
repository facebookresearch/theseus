# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

import theseus as th
from tests.core.common import BATCH_SIZES_TO_TEST
from tests.geometry.test_se2 import create_random_se2
from theseus.utils import numeric_jacobian


def test_error_quasi_static_pushing_planar_se2():

    # c_square is c**2, c = max_torque / max_force is a hyper param dependent on object
    c_square = torch.Tensor([1.0])
    cost_weight = th.ScaleCostWeight(1)

    inputs = {
        "obj1": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, np.pi / 2],
            ]
        ),
        "obj2": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, np.pi / 2],
                [1.0, 1.0, np.pi / 4],
                [1.0, 1.0, np.pi / 4],
                [0.0, 0.0, 0.0],
            ]
        ),
        "eff1": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.pi / 2],
                [0.0, 0.0, np.pi / 4],
                [2.0, 2.0, np.pi / 4],
                [2.0, 2.0, np.pi / 4],
            ]
        ),
        "eff2": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, np.pi],
                [1.0, 1.0, np.pi / 2],
                [3.0, 3.0, np.pi / 2],
                [3.0, 3.0, np.pi / 2],
            ]
        ),
    }

    outputs = {
        "error": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.57079633],
                [0.0, 0.0, -0.78539816],
                [0.0, 2.22144147, -0.785398163],
                [2.71238898, -6.71238898, 1.57079633],
            ]
        )
    }
    n_tests = outputs["error"].shape[0]
    for i in range(0, n_tests):
        obj1 = th.SE2(x_y_theta=(inputs["obj1"][i, :]).unsqueeze(0))
        obj2 = th.SE2(x_y_theta=(inputs["obj2"][i, :]).unsqueeze(0))
        eff1 = th.SE2(x_y_theta=(inputs["eff1"][i, :]).unsqueeze(0))
        eff2 = th.SE2(x_y_theta=(inputs["eff2"][i, :]).unsqueeze(0))

        cost_fn = th.eb.QuasiStaticPushingPlanar(
            obj1, obj2, eff1, eff2, c_square, cost_weight
        )

        actual = cost_fn.error()
        _, actual2 = cost_fn.jacobians()
        expected = outputs["error"][i, :]

        print(
            f"actual: {actual.squeeze().numpy()}, expected: {expected.squeeze().numpy()}"
        )
        assert np.allclose(actual.squeeze().numpy(), expected.squeeze().numpy())
        assert torch.allclose(actual, actual2)


def test_quasi_static_pushing_planar_jacobians():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):  # repeat a bunch of times
        for batch_size in BATCH_SIZES_TO_TEST:
            obj1 = create_random_se2(batch_size, rng)
            obj2 = create_random_se2(batch_size, rng)
            eff1 = create_random_se2(batch_size, rng)
            eff2 = create_random_se2(batch_size, rng)
            c_square = torch.Tensor([1.0])
            cost_weight = th.ScaleCostWeight(1)

            cost_fn = th.eb.QuasiStaticPushingPlanar(
                obj1, obj2, eff1, eff2, c_square, cost_weight
            )
            jacobians, _ = cost_fn.jacobians()

            def new_error_fn(groups):
                new_cost_fn = th.eb.QuasiStaticPushingPlanar(
                    groups[0], groups[1], groups[2], groups[3], c_square, cost_weight
                )
                return th.Vector(tensor=new_cost_fn.error())

            expected_jacs = numeric_jacobian(
                new_error_fn, [obj1, obj2, eff1, eff2], delta_mag=1e-6
            )

            def _check_jacobian(actual_, expected_):
                # This makes failures more explicit than torch.allclose()
                diff = (expected_ - actual_).norm(p=float("inf"))
                assert diff < 1e-5

            for i in range(len(expected_jacs)):
                _check_jacobian(jacobians[i], expected_jacs[i])
