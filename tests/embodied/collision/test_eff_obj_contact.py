# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa
import torch

import theseus as th
from tests.core.common import BATCH_SIZES_TO_TEST
from tests.embodied.collision.utils import (
    random_origin,
    random_scalar,
    random_sdf_data,
)
from tests.geometry.test_se2 import create_random_se2
from theseus.utils import numeric_jacobian


def test_eff_obj_interesect_jacobians():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in BATCH_SIZES_TO_TEST:
        obj = create_random_se2(batch_size, rng)
        eff = create_random_se2(batch_size, rng)
        origin = random_origin(batch_size)
        sdf_data = random_sdf_data(batch_size, 10, 10)
        cell_size = random_scalar(batch_size)
        eff_radius = random_scalar(batch_size)
        cost_weight = th.ScaleCostWeight(1.0)
        cost_function = th.eb.EffectorObjectContactPlanar(
            obj, eff, origin, sdf_data, cell_size, eff_radius, cost_weight
        )
        jacobians, _ = cost_function.jacobians()

        def new_error_fn(groups):
            new_cost_function = th.eb.EffectorObjectContactPlanar(
                groups[0],
                groups[1],
                origin,
                sdf_data,
                cell_size,
                eff_radius,
                cost_weight,
            )
            return th.Vector(tensor=new_cost_function.error())

        expected_jacs = numeric_jacobian(
            new_error_fn, [obj, eff], function_dim=1, delta_mag=1e-6
        )

        def _check_jacobian(actual_, expected_):
            # This makes failures more explicit than torch.allclose()
            diff = (expected_ - actual_).norm(p=float("inf"))
            assert diff < 1e-5

        for i in range(len(expected_jacs)):
            _check_jacobian(jacobians[i], expected_jacs[i])


def _load_sdf_data_from_file(filename):

    with open(filename) as f:
        import json

        sdf_data = json.load(f)
    sdf_data_vec = sdf_data["grid_data"]

    sdf_data_mat = np.zeros((sdf_data["grid_size_y"], sdf_data["grid_size_x"]))
    for i in range(sdf_data_mat.shape[0]):
        for j in range(sdf_data_mat.shape[1]):
            sdf_data_mat[i, j] = sdf_data_vec[i][j]

    sdf_data_mat = torch.Tensor(sdf_data_mat).unsqueeze(0)
    cell_size = torch.Tensor([sdf_data["grid_res"]]).unsqueeze(0)
    origin = torch.Tensor(
        [sdf_data["grid_origin_x"], sdf_data["grid_origin_y"]]
    ).unsqueeze(0)

    return sdf_data_mat, cell_size, origin


def _create_sdf_data(sdf_idx=0):

    cell_size = 0.01
    origin_x, origin_y = 0.0, 0.0
    sdf_data = np.loadtxt(
        open("tests/embodied/collision/sdf_data.csv", "rb"),
        delimiter=",",
        skiprows=0,
    )
    sdf_data_mat = (sdf_data[sdf_idx, :]).reshape(10, 10)

    sdf_data_mat = torch.Tensor(sdf_data_mat).unsqueeze(0)
    cell_size = torch.Tensor([cell_size]).unsqueeze(0)
    origin = torch.Tensor([origin_x, origin_y]).unsqueeze(0)

    return sdf_data_mat, cell_size, origin


def test_eff_obj_interesect_errors():

    eff_radius = torch.Tensor([0.0])
    cost_weight = th.ScaleCostWeight(1.0)

    inputs = {
        "obj": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.02, 0.01, 0.0],
                [0.02, 0.03, 0.0],
            ]
        ),
        "eff": torch.DoubleTensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.pi / 2],
                [0.01, 0.02, 0.0],
                [0.01, 0.02, np.pi / 4],
                [0.02, 0.06, 0.0],
            ]
        ),
    }
    outputs = {
        "error": torch.DoubleTensor(
            [
                [[0.72426180], [0.72426180], [0.17443964], [0.0], [0.67393521]],
                [[0.22422976], [0.22422976], [0.32247697], [0.0], [0.32451792]],
                [[0.45021893], [0.45021893], [0.06058549], [0.0], [0.76042012]],
                [[0.23016018], [0.23016018], [0.33050028], [0.0], [0.81287555]],
                [[0.48738902], [0.48738902], [0.38351342], [0.0], [0.96552167]],
            ]
        )
    }

    for sdf_idx in range(0, 5):
        sdf_data, cell_size, origin = _create_sdf_data(sdf_idx)

        obj = th.SE2(x_y_theta=inputs["obj"])
        eff = th.SE2(x_y_theta=inputs["eff"])

        cost_fn = th.eb.EffectorObjectContactPlanar(
            obj,
            eff,
            origin.repeat(5, 1),
            sdf_data.repeat(5, 1, 1),
            cell_size.repeat(5, 1),
            eff_radius,
            cost_weight,
        )

        actual = cost_fn.error()
        expected = outputs["error"][sdf_idx, :]
        assert torch.allclose(actual, expected)


def test_eff_obj_variable_type():
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(10):
        for batch_size in BATCH_SIZES_TO_TEST:
            obj = create_random_se2(batch_size, rng)
            eff = create_random_se2(batch_size, rng)
            origin = random_origin(batch_size)
            sdf_data = random_sdf_data(batch_size, 10, 10)
            cell_size = random_scalar(batch_size)
            eff_radius = random_scalar(batch_size)
            cost_weight = th.ScaleCostWeight(1.0)
            cost_function = th.eb.EffectorObjectContactPlanar(
                obj, eff, origin, sdf_data, cell_size, eff_radius, cost_weight
            )

            assert isinstance(cost_function.eff_radius, th.Variable)
            if isinstance(eff_radius, th.Variable):
                assert cost_function.eff_radius is eff_radius
            else:
                assert np.allclose(
                    cost_function.eff_radius.tensor.cpu().numpy(), eff_radius
                )

            eff_radius_t = torch.rand(batch_size, 1).double()

            cost_function = th.eb.EffectorObjectContactPlanar(
                obj, eff, origin, sdf_data, cell_size, eff_radius_t, cost_weight
            )

            assert isinstance(cost_function.eff_radius, th.Variable)
            assert np.allclose(cost_function.eff_radius.tensor, eff_radius_t)
            assert len(cost_function.eff_radius.shape) == 2

            eff_radius_f = torch.rand(1)

            cost_function = th.eb.EffectorObjectContactPlanar(
                obj, eff, origin, sdf_data, cell_size, eff_radius_f, cost_weight
            )

            assert isinstance(cost_function.eff_radius, th.Variable)
            assert np.allclose(cost_function.eff_radius.tensor.item(), eff_radius_f)
            assert len(cost_function.eff_radius.shape) == 2
