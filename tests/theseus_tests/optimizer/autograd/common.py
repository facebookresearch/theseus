# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import grad, gradcheck


def check_grad(solve_func, inputs, eps, atol, rtol):
    assert gradcheck(solve_func, inputs, eps=eps, atol=atol)

    A_val, b = inputs[0], inputs[1]
    # Check that the gradient works correctly for floating point data
    out = solve_func(*inputs).sum()
    gA, gb = grad(out, (A_val, b))

    A_float = A_val.float()
    b_float = b.float()
    inputs2 = (A_float, b_float) + inputs[2:]
    out_float = solve_func(*inputs2).sum()
    gA_float, gb_float = grad(out_float, (A_float, b_float))

    # This mostly checks that backward() is not accumulating
    # additional floating point errors on top of those expected by
    # converting the input from double to float. Naive float casting
    # in the backward python ops can result in differences in order of magnitude
    # even for well-conditioned systems. These checks cover that case.
    #
    # Do note that, in general, it's possible to construct very
    # ill-conditioned systems where the initial loss or precision is enough
    # to get large gradient errors no matter what we do. This checks
    # are not meant to capture such scenarios.
    torch.testing.assert_close(gA, gA_float.double(), rtol=rtol, atol=atol)
    torch.testing.assert_close(gb, gb_float.double(), rtol=rtol, atol=atol)
