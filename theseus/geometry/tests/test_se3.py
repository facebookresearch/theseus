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


def _create_random_se3(batch_size, rng):
    tangent_vector_ang = torch.rand(batch_size, 3) - 0.5
    tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
    tangent_vector_ang *= torch.rand(batch_size, 1) * 2 * np.pi - np.pi
    tangent_vector_lin = torch.randn(batch_size, 3)
    tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)
    return th.SE3.exp_map(tangent_vector.double())


def check_SE3_log_map(tangent_vector, atol=EPS):
    g = th.SE3.exp_map(tangent_vector)
    assert torch.allclose(th.SE3.exp_map(g.log_map()).data, g.data, atol=atol)


def test_exp_map():
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= torch.rand(batch_size, 1).double() * 2 * np.pi - np.pi
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 1e-5
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 3e-3
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 2 * np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_exp_map(tangent_vector, th.SE3)


def test_log_map():
    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3) - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= torch.rand(batch_size, 1).double() * 2 * np.pi - np.pi
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 1e-5
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 3e-3
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= 2 * np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)

    for batch_size in [1, 20, 100]:
        tangent_vector_ang = torch.rand(batch_size, 3).double() - 0.5
        tangent_vector_ang /= tangent_vector_ang.norm(dim=1, keepdim=True)
        tangent_vector_ang *= np.pi - 1e-11
        tangent_vector_lin = torch.randn(batch_size, 3).double()
        tangent_vector = torch.cat([tangent_vector_lin, tangent_vector_ang], dim=1)

        check_SE3_log_map(tangent_vector)


def test_adjoint():
    rng = torch.Generator()
    rng.manual_seed(0)
    for batch_size in [1, 20, 100]:
        se3 = _create_random_se3(batch_size, rng)
        tangent = torch.randn(batch_size, 6).double()
        check_adjoint(se3, tangent)
