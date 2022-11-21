# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def check_lie_group_function(module, method: str, atol: float, *args):
    func_call = getattr(module, "_" + method + "_impl")
    func_apply = getattr(module, method)

    grad_call = torch.autograd.functional.jacobian(func_call, args)
    grad_apply = torch.autograd.functional.jacobian(func_apply, args)

    for grad_c, grad_a in zip(grad_call, grad_apply):
        assert torch.allclose(grad_c, grad_a, atol=atol)
