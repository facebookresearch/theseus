# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import pytest
import torch

from torchlie.functional import SE3, SO3, enable_checks


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
@pytest.mark.parametrize("offset", [0.0, 1e-4])
@pytest.mark.parametrize("axis_sign", [-1.0, 1.0])
def test_log_tied_largest_diagonal(dtype, batch_shape, offset, axis_sign):
    # For this axis, diagonal entries 1 and 2 tie above entry 0. Selecting
    # row 0 gives a zero vector instead of the rotation axis near pi.
    axis = torch.tensor([0.0, 1.0, axis_sign], dtype=dtype) / math.sqrt(2)
    angle = math.pi - offset
    skew = torch.tensor(
        [[0.0, -axis_sign, 1.0], [axis_sign, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=dtype,
    ) / math.sqrt(2)
    rotation = (
        math.cos(angle) * torch.eye(3, dtype=dtype)
        + (1 - math.cos(angle)) * torch.outer(axis, axis)
        + math.sin(angle) * skew
    )
    if offset == 0.0:
        rotation = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, axis_sign], [0.0, axis_sign, 0.0]],
            dtype=dtype,
        )
    rotation = rotation.expand(*batch_shape, 3, 3)
    translation = torch.tensor([0.3, -0.5, 0.7], dtype=dtype).expand(*batch_shape, 3)
    transform = torch.cat((rotation, translation.unsqueeze(-1)), dim=-1)
    atol = 2e-6 if dtype == torch.float32 else 1e-10
    with enable_checks():
        for group, value in [(SO3, rotation), (SE3, transform)]:
            actual = group.log(value)
            jacobians = []
            with_jacobians = group.log(value, jacobians=jacobians)
            assert torch.isfinite(actual).all()
            assert torch.isfinite(jacobians[0]).all()
            torch.testing.assert_close(actual, with_jacobians)
            torch.testing.assert_close(group.exp(actual), value, atol=atol, rtol=atol)
            angular = actual[..., -3:]
            # The logarithm at pi has two valid axis signs.
            torch.testing.assert_close(
                torch.linalg.vector_norm(angular, dim=-1),
                torch.full(batch_shape, angle, dtype=dtype),
                atol=atol,
                rtol=atol,
            )
            if offset:
                torch.testing.assert_close(
                    angular, (angle * axis).expand_as(angular), atol=atol, rtol=atol
                )
