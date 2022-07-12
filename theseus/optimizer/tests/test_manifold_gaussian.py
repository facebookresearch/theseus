# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th


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
        precision_sqrt = torch.rand(mean[0].shape[0], dof, dof)
        precision = torch.bmm(precision_sqrt, precision_sqrt.transpose(1, 2))
        precision += torch.eye(dof)[None, ...].repeat(mean[0].shape[0], 1, 1)
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

        t = th.ManifoldGaussian(mean, precision=precision, name=name)
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
            precision = torch.eye(dof).to(dtype=mean[0].dtype, device=mean[0].device)
            precision = precision[None, ...].repeat(mean[0].shape[0], 1, 1)
            assert torch.isclose(t.precision, precision).all()

    assert len(set(all_ids)) == len(all_ids)


def test_to():
    for i in range(10):
        mean, precision = random_manifold_gaussian_params()
        t = th.ManifoldGaussian(mean, precision=precision)
        dtype = torch.float64 if np.random.random() < 0.5 else torch.float32
        t.to(dtype)

        for var in t.mean:
            assert var.dtype == dtype
        assert t.precision.dtype == dtype


def test_copy():
    for i in range(10):
        mean, precision = random_manifold_gaussian_params()
        n_vars = len(mean)
        var = th.ManifoldGaussian(mean, precision, name="var")

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

        var = th.ManifoldGaussian(mean, precision, name="var")

        # check update
        new_mean_good = []
        for j in range(n_vars):
            new_var = mean[j].__class__.rand(batch_size)
            new_mean_good.append(new_var)
        new_precision_good = torch.eye(dof)[None, ...].repeat(batch_size, 1, 1)

        var.update(new_mean_good, new_precision_good)

        assert var.precision is new_precision_good
        for j in range(n_vars):
            assert torch.allclose(var.mean[j].tensor, new_mean_good[j].tensor)

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


def test_local_gaussian():
    manif_types = [th.Point2, th.Point3, th.SE2, th.SE3, th.SO2, th.SO3]

    for i in range(50):
        batch_size = np.random.randint(1, 100)
        ix = np.random.randint(len(manif_types))
        mean = [manif_types[ix].rand(batch_size)]
        precision = torch.eye(mean[0].dof())[None, ...].repeat(batch_size, 1, 1)
        gaussian = th.ManifoldGaussian(mean, precision)
        variable = manif_types[ix].rand(batch_size)

        mean_tp, lam_tp1 = th.local_gaussian(variable, gaussian, return_mean=True)
        eta_tp, lam_tp2 = th.local_gaussian(variable, gaussian, return_mean=False)

        assert torch.allclose(lam_tp1, lam_tp2)

        # check mean and eta are consistent
        mean_tp_calc = torch.matmul(lam_tp1, mean_tp.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(eta_tp, mean_tp_calc)

        # check raises error if gaussian over mulitple Manifold objects
        bad_mean = mean + [variable]
        dof = sum([var.dof() for var in bad_mean])
        precision = torch.zeros(batch_size, dof, dof)
        bad_gaussian = th.ManifoldGaussian(bad_mean, precision)
        with pytest.raises(ValueError):
            _, _ = th.local_gaussian(variable, bad_gaussian, return_mean=True)

        # check raises error if gaussian over mulitple Manifold objects
        bad_ix = np.mod(ix + 1, len(manif_types))
        bad_variable = manif_types[bad_ix].rand(batch_size)
        with pytest.raises(ValueError):
            _, _ = th.local_gaussian(bad_variable, gaussian, return_mean=True)


def test_retract_gaussian():
    manif_types = [th.Point2, th.Point3, th.SE2, th.SE3, th.SO2, th.SO3]

    for i in range(50):
        batch_size = np.random.randint(1, 100)
        ix = np.random.randint(len(manif_types))
        variable = manif_types[ix].rand(batch_size)

        mean_tp = torch.rand(batch_size, variable.dof())
        lam_tp = torch.eye(variable.dof())[None, ...].repeat(batch_size, 1, 1)

        gaussian = th.retract_gaussian(variable, mean_tp, lam_tp)
        assert torch.allclose(gaussian.mean[0].tensor, variable.retract(mean_tp).tensor)
