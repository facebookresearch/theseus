# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

import theseus.labs.lie as lie
import theseus.labs.lie.functional.se3_impl as se3_impl
import theseus.labs.lie.functional.so3_impl as so3_impl


def _get_impl(ltype):
    return {lie.SE3: se3_impl, lie.SO3: so3_impl}[ltype]


@pytest.mark.parametrize("op_name", ["log", "inv", "adj"])
@pytest.mark.parametrize("ltype", [lie.SE3, lie.SO3])
@pytest.mark.parametrize("batch_size", [5])
def test_op_no_args(op_name, ltype, batch_size):
    impl_name = {"log": "log", "inv": "inverse", "adj": "adjoint"}[op_name]
    out_is_group = op_name == "inv"
    rng = torch.Generator()
    rng.manual_seed(0)
    impl_module = _get_impl(ltype)
    x = lie.rand(ltype, batch_size, generator=rng)
    out = getattr(x, f"{op_name}")()
    out = out._t if out_is_group else out
    impl_out = getattr(impl_module, f"_{impl_name}_autograd_fn")(x._t)
    torch.testing.assert_close(out, impl_out)

    if op_name == "log":
        out1, out2 = getattr(x, f"j{op_name}")()
        impl_out1, impl_out2 = getattr(impl_module, f"_j{impl_name}_autograd_fn")(x._t)
        impl_out2 = impl_out2._t if out_is_group else impl_out2
        torch.testing.assert_close(out1, impl_out1)
        torch.testing.assert_close(out2, impl_out2)
