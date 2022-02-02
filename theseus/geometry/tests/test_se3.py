# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch

import theseus as th

from .common import check_adjoint


def _create_random_se3(batch_size, rng):
    x_y_z_quaternion = torch.rand(batch_size, 7, generator=rng).double() - 0.5
    quaternion_norm = torch.linalg.norm(x_y_z_quaternion[:, 3:], dim=1, keepdim=True)
    x_y_z_quaternion[:, 3:] /= quaternion_norm

    return th.SE3(x_y_z_quaternion=x_y_z_quaternion)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3 = _create_random_se3(batch_size, rng)
        tangent = torch.randn(batch_size, 6).double()
        check_adjoint(se3, tangent)
