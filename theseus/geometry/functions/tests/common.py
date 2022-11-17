# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def check_lie_group_function(func, method: str, atol: float, *args):
    def func_call(*args):
        return getattr(func, method).call(*args)

    def func_apply(*args):
        return getattr(func, method).apply(*args)

    grad_call = torch.autograd.functional.jacobian(func_call, args)
    grad_apply = torch.autograd.functional.jacobian(func_apply, args)

    for grad_c, grad_a in zip(grad_call, grad_apply):
        assert torch.allclose(grad_c, grad_a, atol=atol)
