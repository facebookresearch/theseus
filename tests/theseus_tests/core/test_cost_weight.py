# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest  # noqa: F401
import torch

import theseus as th

from .common import (
    BATCH_SIZES_TO_TEST,
    check_another_theseus_function_is_copy,
    check_another_theseus_tensor_is_copy,
)


def test_copy_scale_cost_weight():
    scale = th.Variable(torch.tensor(1.0))
    p1 = th.ScaleCostWeight(scale, name="scale_cost_weight")
    for the_copy in [p1.copy(), copy.deepcopy(p1)]:
        check_another_theseus_function_is_copy(p1, the_copy, new_name=f"{p1.name}_copy")
        check_another_theseus_tensor_is_copy(scale, the_copy.scale)


def test_copy_diagonal_cost_weight():
    diagonal = th.Variable(torch.ones(1, 3))
    p1 = th.DiagonalCostWeight(diagonal, name="diagonal_cost_weight")
    for the_copy in [p1.copy(), copy.deepcopy(p1)]:
        check_another_theseus_function_is_copy(p1, the_copy, new_name=f"{p1.name}_copy")
        check_another_theseus_tensor_is_copy(diagonal, the_copy.diagonal)


def test_scale_cost_weight():
    for dim in BATCH_SIZES_TO_TEST:
        for batch_size in BATCH_SIZES_TO_TEST:
            v1 = th.Vector(tensor=torch.ones(batch_size, dim))
            z = th.Vector(tensor=torch.zeros(batch_size, dim))
            scale = torch.randn(1).item()
            expected_err = torch.ones(batch_size, dim) * scale
            expected_jac = (torch.eye(dim).unsqueeze(0) * scale).expand(
                batch_size, dim, dim
            )

            def _check(cw):
                cf1 = th.Difference(v1, z, cw)
                jacobians, error = cf1.weighted_jacobians_error()
                assert error.allclose(expected_err)
                assert jacobians[0].allclose(expected_jac)

            _check(th.ScaleCostWeight(scale))
            _check(th.ScaleCostWeight(torch.ones(1) * scale))
            _check(th.ScaleCostWeight(torch.ones(batch_size) * scale))
            _check(th.ScaleCostWeight(torch.ones(1, 1) * scale))
            _check(th.ScaleCostWeight(torch.ones(batch_size, 1) * scale))
            _check(th.ScaleCostWeight(th.Variable(torch.ones(1, 1) * scale)))

            batched_scale = torch.randn(batch_size, 1)
            expected_err = batched_scale * torch.ones(batch_size, dim)
            expected_jac = (torch.eye(dim).unsqueeze(0)).expand(
                batch_size, dim, dim
            ) * batched_scale.view(-1, 1, 1)

            cf1 = th.Difference(v1, z, th.ScaleCostWeight(batched_scale))
            jacobians, error = cf1.weighted_jacobians_error()
            assert error.allclose(expected_err)
            assert jacobians[0].allclose(expected_jac)


def test_diagonal_cost_weight():
    for dim in BATCH_SIZES_TO_TEST:
        for batch_size in BATCH_SIZES_TO_TEST:
            v1 = th.Vector(tensor=torch.ones(batch_size, dim))
            z = th.Vector(tensor=torch.zeros(batch_size, dim))
            diagonal = torch.randn(dim)
            d_matrix = diagonal.diag().unsqueeze(0)
            expected_err = (d_matrix @ torch.ones(batch_size, dim, 1)).view(
                batch_size, dim
            )
            expected_jac = d_matrix @ (torch.eye(dim).unsqueeze(0)).expand(
                batch_size, dim, dim
            )

            def _check(cw):
                cf1 = th.Difference(v1, z, cw)
                jacobians, error = cf1.weighted_jacobians_error()
                assert error.allclose(expected_err)
                assert jacobians[0].allclose(expected_jac)

            diagonal = diagonal.unsqueeze(0)  # add batch dimension
            _check(th.DiagonalCostWeight(diagonal.tolist()))
            _check(th.DiagonalCostWeight(diagonal))
            _check(th.DiagonalCostWeight(th.Variable(diagonal)))

            batched_diagonal = torch.randn(batch_size, dim)
            d_matrix = batched_diagonal.diag_embed()
            expected_err = (d_matrix @ torch.ones(batch_size, dim, 1)).view(
                batch_size, dim
            )
            expected_jac = d_matrix @ (torch.eye(dim).unsqueeze(0)).expand(
                batch_size, dim, dim
            )

            cf1 = th.Difference(v1, z, th.DiagonalCostWeight(batched_diagonal))
            jacobians, error = cf1.weighted_jacobians_error()
            assert error.allclose(expected_err)
            assert jacobians[0].allclose(expected_jac)
