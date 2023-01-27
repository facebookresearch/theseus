# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

import theseus.labs.lie as lie
import theseus.labs.lie.functional.se3_impl as se3_impl
import theseus.labs.lie.functional.so3_impl as so3_impl


def _get_impl(ltype):
    return {lie.SE3: se3_impl, lie.SO3: so3_impl}[ltype]


def test_log():
    rng = torch.Generator()
    rng.manual_seed(0)
    for ltype in [lie.SE3, lie.SO3]:
        impl_module = _get_impl(ltype)
        x = lie.rand(ltype, 20, generator=rng)
        out = x.log()
        impl_out = impl_module._log_autograd_fn(x._t)
        torch.testing.assert_close(out, impl_out)

        out1, out2 = x.jlog()
        impl_out1, impl_out2 = impl_module._jlog_autograd_fn(x._t)
        torch.testing.assert_close(out1, impl_out1)
        torch.testing.assert_close(out2, impl_out2)
