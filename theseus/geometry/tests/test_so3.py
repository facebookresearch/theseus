# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th
from theseus.constants import EPS

from .common import check_adjoint, check_exp_map


def _create_random_so3(batch_size, rng):
    q = torch.rand(batch_size, 4, generator=rng).double() - 0.5
    qnorm = torch.linalg.norm(q, dim=1, keepdim=True)
    q = q / qnorm
    return th.SO3(quaternion=q)


def check_SO3_log_map(tangent_vector, group_cls):
    error = (tangent_vector - group_cls.exp_map(tangent_vector).log_map()).norm(dim=1)
    error = torch.minimum(error, (error - 2 * np.pi).abs())
    assert torch.allclose(error, torch.zeros_like(error), atol=EPS)


def test_exp_map():
    for batch_size in [1, 20, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-5
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        check_exp_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        check_exp_map(tangent_vector, th.SO3)


def test_log_map():
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        check_SO3_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-6
        check_SO3_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-3
        check_SO3_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-11
        check_SO3_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 1e-3
        check_SO3_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 2 * np.pi - 1e-11
        check_SO3_log_map(tangent_vector, th.SO3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3 = _create_random_so3(batch_size, rng)
        tangent = torch.randn(batch_size, 3).double()
        check_adjoint(so3, tangent)
