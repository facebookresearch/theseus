# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th

from .common import MockVar


def test_variable_init():
    all_ids = []
    for i in range(100):
        if np.random.random() < 0.5:
            name = f"name_{i}"
        else:
            name = None
        data = torch.rand(1, 1)
        t = th.Variable(data, name=name)
        all_ids.append(t._id)
        if name is not None:
            assert name == t.name
        else:
            assert t.name == f"Variable__{t._id}"
        assert t.tensor.allclose(data)

    assert len(set(all_ids)) == len(all_ids)


def test_properties():
    for i in range(100):
        length = np.random.randint(2, 5)
        batch_size = np.random.randint(1, 100)
        dtype = torch.float if np.random.random() < 0.5 else torch.long
        data = torch.ones(batch_size, length, dtype=dtype)
        t = th.Variable(data)
        assert t.shape == data.shape
        assert t.ndim == data.ndim
        assert t.dtype == dtype


def test_update():
    for _ in range(10):
        for length in range(1, 10):
            var = MockVar(length)
            batch_size = np.random.randint(1, 10)
            # check update from torch tensor
            new_data_good = torch.rand(batch_size, length)
            var.update(new_data_good)
            assert var.tensor is new_data_good
            # check update from variable
            new_data_good_wrapped = torch.rand(batch_size, length)
            another_var = MockVar(length, tensor=new_data_good_wrapped)
            var.update(another_var)
            # check raises error on shape
            new_data_bad = torch.rand(batch_size, length + 1)
            with pytest.raises(ValueError):
                var.update(new_data_bad)
            # check raises error on dtype
            new_data_bad = torch.rand(batch_size, length + 1).double()
            with pytest.raises(ValueError):
                var.update(new_data_bad)
            # check batch indices to ignore
            how_many = np.random.randint(1, batch_size + 1)
            ignore_indices = np.random.choice(batch_size, size=how_many)
            ignore_mask = torch.zeros(batch_size).bool()
            ignore_mask[ignore_indices] = 1
            old_data = var.tensor.clone()
            new_data_some_ignored = torch.randn(batch_size, length)
            if ignore_indices[0] % 2 == 0:  # randomly wrap into a variable to also test
                new_data_some_ignored = MockVar(length, new_data_some_ignored)
            var.update(new_data_some_ignored, batch_ignore_mask=ignore_mask)
            for i in range(batch_size):
                if ignore_mask[i] == 1:
                    assert torch.allclose(var[i], old_data[i])
                else:
                    if isinstance(new_data_some_ignored, th.Variable):
                        new_data_some_ignored = new_data_some_ignored.tensor
                    assert torch.allclose(var[i], new_data_some_ignored[i])
