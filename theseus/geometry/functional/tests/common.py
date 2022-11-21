# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def check_lie_group_function(module, func_name: str, atol: float, *args):
    func_impl = getattr(module, "_" + func_name + "_impl")
    func = getattr(module, func_name)

    jacs_impl = torch.autograd.functional.jacobian(func_impl, args)
    jacs = torch.autograd.functional.jacobian(func, args)

    for jac_impl, jac in zip(jacs_impl, jacs):
        assert torch.allclose(jac_impl, jac, atol=atol)
