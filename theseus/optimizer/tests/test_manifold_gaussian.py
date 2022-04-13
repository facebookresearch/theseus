# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus import ManifoldGaussian


def random_manifold_gaussian_params():
    manif_types = [th.Point2, th.Point3, th.SE2, th.SE3, th.SO2, th.SO3]

    n_vars = np.random.randint(1, 5)
    batch_size = np.random.randint(1, 100)
    mean = []
    dof = 0
    for i in range(n_vars):
        ix = np.random.randint(len(manif_types))
        var = manif_types[ix].rand(batch_size)
        mean.append(var)
        dof += var.dof()

    if np.random.random() < 0.5:
        precision = torch.zeros(mean[0].shape[0], dof, dof)
    else:
        precision = None

    return mean, precision


def test_init():
    all_ids = []
    for i in range(10):
        if np.random.random() < 0.5:
            name = f"name_{i}"
        else:
            name = None
        mean, precision = random_manifold_gaussian_params()
        dof = sum([v.dof() for v in mean])
        n_vars = len(mean)

        t = ManifoldGaussian(mean, precision=precision, name=name)
        all_ids.append(t._id)
        if name is not None:
            assert name == t.name
        else:
            assert t.name == f"ManifoldGaussian__{t._id}"
        assert t.dof == dof
        for j in range(n_vars):
            assert t.mean[j] == mean[j]
        if precision is not None:
            assert torch.isclose(t.precision, precision).all()
        else:
            precision = torch.zeros(mean[0].shape[0], dof, dof).to(
                dtype=mean[0].dtype, device=mean[0].device
            )
            assert torch.isclose(t.precision, precision).all()

    assert len(set(all_ids)) == len(all_ids)


def test_to():
    for i in range(10):
        mean, precision = random_manifold_gaussian_params()
        t = ManifoldGaussian(mean, precision=precision)
        dtype = torch.float64 if np.random.random() < 0.5 else torch.long
        t.to(dtype)

        for var in t.mean:
            assert var.dtype == dtype
        assert t.precision.dtype == dtype


def test_copy():
    for i in range(10):
        mean, precision = random_manifold_gaussian_params()
        n_vars = len(mean)
        var = ManifoldGaussian(mean, precision, name="var")

        var.name = "old"
        new_var = var.copy(new_name="new")
        assert var is not new_var
        for j in range(n_vars):
            assert var.mean[j] is not new_var.mean[j]
        assert var.precision is not new_var.precision
        assert torch.allclose(var.precision, new_var.precision)
        assert new_var.name == "new"
        new_var_no_name = copy.deepcopy(var)
        assert new_var_no_name.name == f"{var.name}_copy"


def test_update():
    for i in range(10):
        mean, precision = random_manifold_gaussian_params()
        n_vars = len(mean)
        dof = sum([v.dof() for v in mean])
        batch_size = mean[0].shape[0]

        var = ManifoldGaussian(mean, precision, name="var")

        # check update
        new_mean_good = []
        for j in range(n_vars):
            new_var = mean[j].__class__.rand(batch_size)
            new_mean_good.append(new_var)
        new_precision_good = torch.eye(dof)[None, ...].repeat(batch_size, 1, 1)

        var.update(new_mean_good, new_precision_good)

        assert var.precision is new_precision_good
        for j in range(n_vars):
            assert torch.allclose(var.mean[j].data, new_mean_good[j].data)

        # check raises error on shape for precision
        new_precision_bad = torch.eye(dof + 1)[None, ...].repeat(batch_size, 1, 1)
        with pytest.raises(ValueError):
            var.update(new_mean_good, new_precision_bad)

        # check raises error on dtype for precision
        new_precision_bad = torch.eye(dof)[None, ...].repeat(batch_size, 1, 1).double()
        with pytest.raises(ValueError):
            var.update(new_mean_good, new_precision_bad)

        # check raises error for non symmetric precision
        new_precision_bad = torch.eye(dof)[None, ...].repeat(batch_size, 1, 1)
        if dof > 1:
            new_precision_bad[0, 1, 0] += 1.0
            with pytest.raises(ValueError):
                var.update(new_mean_good, new_precision_bad)

        # check raises error on wrong number of mean variables
        new_mean_bad = new_mean_good[:-1]
        with pytest.raises(ValueError):
            var.update(new_mean_bad, new_precision_good)

        # check raises error on wrong variable type
        new_mean_bad = new_mean_good
        new_mean_bad[-1] = th.Vector(10)
        with pytest.raises(ValueError):
            var.update(new_mean_bad, new_precision_good)
