# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS
from tests.core.common import check_copy_var

from .common import (
    BATCH_SIZES_TO_TEST,
    check_jacobian_for_local,
    check_projection_for_exp_map,
    check_projection_for_log_map,
)

torch.manual_seed(0)


def test_item():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            assert torch.allclose(v1.tensor, t1)
            v1[0, 0] = 11.1
            assert not torch.allclose(v1.tensor, t1)
            t1[0, 0] = 11.1
            assert torch.allclose(v1.tensor, t1)


def test_add():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            t2 = torch.rand(i, j)
            v2 = th.Vector(tensor=t2.clone(), name="v2")
            vsum = th.Vector(tensor=t1 + t2)
            assert (v1 + v2).allclose(vsum)
            assert v1.compose(v2).allclose(vsum)


def test_sub():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            t2 = torch.rand(i, j)
            v2 = th.Vector(tensor=t2.clone(), name="v2")
            assert (v1 - v2).allclose(th.Vector(tensor=t1 - t2))
            v2 = -v2
            assert (v1 + v2).allclose(th.Vector(tensor=t1 - t2))


def test_mul():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            assert (v1 * torch.tensor(2.1)).allclose(th.Vector(tensor=t1 * 2.1))
            assert (torch.tensor(1.1) * v1).allclose(th.Vector(tensor=t1 * 1.1))


def test_div():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            assert (v1 / torch.tensor(2.1)).allclose(th.Vector(tensor=t1 / 2.1))


def test_matmul():
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                t = torch.rand(i, j, k)
                t1 = torch.rand(i, j)
                v1 = th.Vector(tensor=t1.clone(), name="v1")
                v1t = v1 @ t
                assert v1t.allclose((t1.unsqueeze(1) @ t).squeeze(1))
                assert v1t.shape == (i, k)
                t2 = torch.rand(i, k)
                v2 = th.Vector(tensor=t2.clone(), name="v2")
                tv2 = t @ v2
                assert tv2.allclose((t @ t2.unsqueeze(2)).squeeze(2))
                assert tv2.shape == (i, j)


def test_dot():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            t2 = torch.rand(i, j)
            v2 = th.Vector(tensor=t2.clone(), name="v2")
            assert torch.allclose(v1.dot(v2), torch.mul(t1, t2).sum(-1))
            assert torch.allclose(v1.inner(v2), torch.mul(t1, t2).sum(-1))


def test_outer():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            t2 = torch.rand(i, j)
            v2 = th.Vector(tensor=t2.clone(), name="v2")
            assert torch.allclose(
                v1.outer(v2), torch.matmul(t1.unsqueeze(2), t2.unsqueeze(1))
            )


def test_abs():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            assert v1.abs().allclose(th.Vector(tensor=t1.abs()))


def test_norm():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            assert v1.norm() == t1.norm()
            assert v1.norm(p="fro") == torch.norm(t1, p="fro")


def test_cat():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            t2 = torch.rand(i, j)
            v2 = th.Vector(tensor=t2.clone(), name="v2")
            t3 = torch.rand(i, j)
            v3 = th.Vector(tensor=t3.clone(), name="v3")
            assert v1.cat(v2).allclose(th.Vector(tensor=torch.cat((t1, t2), 1)))
            assert v1.cat((v2, v3)).allclose(
                th.Vector(tensor=torch.cat((t1, t2, t3), 1))
            )


def test_local():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            t2 = torch.rand(i, j)
            v2 = th.Vector(tensor=t2.clone(), name="v2")
            assert torch.allclose(v1._local_impl(v2), t2 - t1)
            assert torch.allclose(v1.local(v2), t2 - t1)
            assert torch.allclose(th.local(v1, v2), t2 - t1)


def test_retract():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v1 = th.Vector(tensor=t1.clone(), name="v1")
            d = torch.rand(i, j)
            assert v1._retract_impl(d).allclose(th.Vector(tensor=t1 + d))
            assert v1.retract(d).allclose(th.Vector(tensor=t1 + d))
            assert th.retract(v1, d).allclose(th.Vector(tensor=t1 + d))


def test_update():
    for dof in range(1, 21):
        v = th.Vector(dof)
        for _ in range(100):
            rng = np.random.default_rng()
            batch_size = rng.integers(low=1, high=100)
            data = torch.rand(batch_size, dof)
            v.update(data)
            assert torch.allclose(data, v.tensor)


def test_copy():
    for i in range(1, 4):
        for j in range(1, 5):
            t1 = torch.rand(i, j)
            v = th.Vector(tensor=t1.clone(), name="v")
            check_copy_var(v)


def test_exp_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in BATCH_SIZES_TO_TEST:
        dim = torch.randint(1, 10, size=[1], generator=rng)[0]
        tangent_vector = torch.rand(batch_size, dim, generator=rng).double() - 0.5
        ret = th.Vector.exp_map(tangent_vector)

        assert torch.allclose(ret.tensor, tangent_vector, atol=EPS)
        check_projection_for_exp_map(
            tangent_vector, Group=th.Vector, is_projected=False
        )


def test_log_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in BATCH_SIZES_TO_TEST:
        dim = torch.randint(1, 10, size=[1], generator=rng)[0]
        group = th.Vector.rand(batch_size, dim, generator=rng)
        ret = group.log_map()

        assert torch.allclose(ret, group.tensor, atol=EPS)
        check_projection_for_log_map(
            tangent_vector=ret, Group=th.Vector, is_projected=False
        )


def test_local_map():
    rng = torch.Generator()
    rng.manual_seed(0)

    for batch_size in BATCH_SIZES_TO_TEST:
        dim = torch.randint(1, 10, size=[1], generator=rng)[0]
        group0 = th.Vector.rand(batch_size, dim, generator=rng)
        group1 = th.Vector.rand(batch_size, dim, generator=rng)

        check_jacobian_for_local(group0, group1, Group=th.Vector, is_projected=False)
