# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th

from .common import MockVar, check_copy_var


def test_copy():
    for i in range(100):
        size = torch.randint(low=1, high=21, size=(1,)).item()
        data = torch.rand(size=(1,) + (size,))
        var = MockVar(size, tensor=data, name="var")
        check_copy_var(var)


def test_var_shape():
    for sz in range(100):
        data = torch.ones(1, sz)
        var = MockVar(sz, tensor=data)
        assert data.shape == var.shape


def test_update_shape_check():
    for sz in range(2, 100):
        data = torch.ones(sz)
        var = MockVar(sz)
        with pytest.raises(ValueError):
            var.update(data)  # no batch dimension
        with pytest.raises(ValueError):
            var.update(data.view(-1, 1))  # wrong dimension
        var.update(data.view(1, -1))
        assert torch.isclose(var.tensor.squeeze(), data).all()


class MockVarNoArgs(th.Manifold):
    def __init__(self, tensor=None, name=None):
        super().__init__(tensor=tensor, name=name)

    @staticmethod
    def _init_tensor():
        return torch.ones(1, 1)

    @staticmethod
    def _check_tensor_impl(tensor: torch.Tensor) -> bool:
        return True

    @staticmethod
    def normalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def dof(self):
        return 0

    def numel(self):
        return 0

    def _local_impl(self, variable2):
        pass

    def _local_jacobian(self, var2):
        pass

    def _retract_impl(self, delta):
        pass

    def _copy_impl(self):
        return MockVarNoArgs()

    def _project_impl(
        self, euclidean_grad: torch.Tensor, is_sparse: bool = False
    ) -> torch.Tensor:
        return euclidean_grad.clone()


def test_variable_no_args_init():
    var = MockVarNoArgs(name="mock")
    assert var.tensor.allclose(torch.ones(1, 1))
    assert var.name == "mock"
    var = MockVarNoArgs(tensor=torch.ones(2, 1))
    assert var.tensor.allclose(torch.ones(2, 1))
    var.update(torch.ones(3, 1))
    assert var.tensor.allclose(torch.ones(3, 1))
