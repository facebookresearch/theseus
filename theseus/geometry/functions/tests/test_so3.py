# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from theseus.geometry.functions.tests.common import check_lie_group_function
from theseus.geometry.functions import SO3Function
from theseus.constants import TEST_EPS

import torch


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp_map(batch_size: int, dtype: torch.dtype):
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype)
    check_lie_group_function(SO3Function, "ExpMap", TEST_EPS, tangent_vector)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adjoint(batch_size: int, dtype: torch.dtype):
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype)
    group = SO3Function.exp_map(tangent_vector)
    check_lie_group_function(SO3Function, "Adjoint", TEST_EPS, group)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(batch_size: int, dtype: torch.dtype):
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype)
    group = SO3Function.exp_map(tangent_vector)
    check_lie_group_function(SO3Function, "Inverse", TEST_EPS, group)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compose(batch_size: int, dtype: torch.dtype):
    tangent_vector0 = torch.rand(batch_size, 3, dtype=dtype)
    tangent_vector1 = torch.rand(batch_size, 3, dtype=dtype)
    group0 = SO3Function.exp_map(tangent_vector0)
    group1 = SO3Function.exp_map(tangent_vector1)
    check_lie_group_function(SO3Function, "Compose", TEST_EPS, group0, group1)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hat(batch_size: int, dtype: torch.dtype):
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype)
    check_lie_group_function(SO3Function, "Hat", TEST_EPS, tangent_vector)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype)
    hat_matrix = SO3Function.hat(tangent_vector)
    check_lie_group_function(SO3Function, "Vee", TEST_EPS, hat_matrix)