# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchlie import reset_global_params, set_global_params
from torchlie.functional import SE3, SO3, enable_checks


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_global_options(dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    g = SE3.rand(1, generator=rng, dtype=getattr(torch, dtype))
    r1 = SE3.log(g)
    set_global_params({f"so3_near_zero_eps_{dtype}": 100.0})
    r2 = SE3.log(g)
    assert not torch.allclose(r1, r2)
    set_global_params({f"so3_near_pi_eps_{dtype}": 100.0})
    r3 = SE3.log(g)

    assert not torch.allclose(r2, r3)
    with enable_checks():
        fake_hat_input = torch.randn(4, 4, dtype=getattr(torch, dtype))
        with pytest.raises(ValueError):
            SE3.check_hat_tensor(fake_hat_input)
        set_global_params({f"so3_hat_eps_{dtype}": 1000.0})
        set_global_params({f"se3_hat_eps_{dtype}": 1000.0})
        SE3.check_hat_tensor(fake_hat_input)

        fake_so3_matrix = torch.randn(3, 3, dtype=getattr(torch, dtype))
        with pytest.raises(ValueError):
            SO3.check_group_tensor(fake_so3_matrix)
        set_global_params({f"so3_matrix_eps_{dtype}": 1000.0})
        SO3.check_group_tensor(fake_so3_matrix)

    with enable_checks():
        set_global_params({f"so3_quat_eps_{dtype}": 0.0})
        fake_hat_input = torch.randn(4, dtype=getattr(torch, dtype))
        with pytest.raises(ValueError):
            SO3.check_unit_quaternion(fake_hat_input)
        set_global_params({f"so3_quat_eps_{dtype}": 1000.0})
        SO3.check_unit_quaternion(fake_hat_input)

    reset_global_params()
