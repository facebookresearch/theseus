# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import theseus as th

from .common import check_adjoint, check_exp_map, check_log_map


def _create_random_so3(batch_size, rng):
    q = torch.rand(batch_size, 4, generator=rng).double() - 0.5
    qnorm = torch.linalg.norm(q, dim=1, keepdim=True)
    q = q / qnorm
    return th.SO3(quaternion=q)


def test_exp_map():
    for batch_size in [1, 20, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        check_exp_map(tangent_vector, th.SO3)


def test_log_map():
    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        check_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= 1e-5
        check_log_map(tangent_vector, th.SO3)

    for batch_size in [1, 2, 100]:
        tangent_vector = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector /= torch.linalg.norm(tangent_vector, dim=1, keepdim=True)
        tangent_vector *= np.pi - 0.1
        check_log_map(tangent_vector, th.SO3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3 = _create_random_so3(batch_size, rng)
        tangent = torch.randn(batch_size, 3).double()
        check_adjoint(so3, tangent)
