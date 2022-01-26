# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th

from .common import (  # check_inverse,; check_compose,; check_exp_map,; check_log_map,
    check_adjoint,
)


def _create_random_so3(batch_size, rng):
    q = torch.rand(batch_size, 4, generator=rng).double() - 0.5
    qnorm = torch.linalg.norm(q, dim=1, keepdim=True)
    q = q / qnorm
    return th.SO3(quaternion=q)


# def test_inverse():
#     rng = torch.Generator()
#     for batch_size in [1, 20, 100]:
#         so3 = _create_random_so3(batch_size, rng)
#         check_inverse(so3)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        so3 = _create_random_so3(batch_size, rng)
        tangent = torch.randn(batch_size, 3).double()
        check_adjoint(so3, tangent)
