import pytest  # noqa: F401
import torch

import theseus as th

from .common import MockVar, check_copy_var


def test_copy():
    for i in range(100):
        size = torch.randint(low=1, high=21, size=(1,)).item()
        data = torch.rand(size=(1,) + (size,))
        var = MockVar(size, data=data, name="var")
        check_copy_var(var)


def test_var_shape():
    for sz in range(100):
        data = torch.ones(1, sz)
        var = MockVar(sz, data=data)
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
        assert torch.isclose(var.data.squeeze(), data).all()


class MockVarNoArgs(th.Manifold):
    def __init__(self, data=None, name=None):
        super().__init__(data=data, name=name)

    @staticmethod
    def _init_data():
        return torch.ones(1, 1)

    def dof(self):
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
    assert var.data.allclose(torch.ones(1, 1))
    assert var.name == "mock"
    var = MockVarNoArgs(data=torch.ones(2, 1))
    assert var.data.allclose(torch.ones(2, 1))
    var.update(torch.ones(3, 1))
    assert var.data.allclose(torch.ones(3, 1))
