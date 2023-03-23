#!/usr/bin/env python3
#
# Copyright 2019 Intel AI, CMU, Bosch AI

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from torch.autograd import Function, Variable, grad
from torch.nn import Module

import numpy as np
import numpy.random as npr

from semantic_version import Version

version = Version(".".join(torch.__version__.split(".")[:3]))
old_torch = version < Version("0.4.0")


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()


class LML(Module):
    def __init__(self, N, eps=1e-4, n_iter=100, branch=None, verbose=0):
        super().__init__()
        self.N = N
        self.eps = eps
        self.n_iter = n_iter
        self.branch = branch
        self.verbose = verbose

    def forward(self, x):
        return LML_Function.apply(
            x, self.N, self.eps, self.n_iter, self.branch, self.verbose
        )


class LML_Function(Function):
    @staticmethod
    def forward(ctx, x, N, eps, n_iter, branch, verbose):
        ctx.N = N
        ctx.eps = eps
        ctx.n_iter = n_iter
        ctx.branch = branch
        ctx.verbose = verbose

        branch = ctx.branch
        if branch is None:
            if not x.is_cuda:
                branch = 10
            else:
                branch = 100

        single = x.ndimension() == 1
        orig_x = x
        if single:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= ctx.N:
            y = (1.0 - 1e-5) * torch.ones(n_batch, nx).type_as(x)
            if single:
                y = y.squeeze(0)
            if old_torch:
                ctx.save_for_backward(orig_x)
                ctx.y = y
                ctx.nu = torch.Tensor()
            else:
                ctx.save_for_backward(orig_x, y, torch.Tensor())
            return y

        x_sorted, _ = torch.sort(x, dim=1, descending=True)

        # The sigmoid saturates the interval [-7, 7]
        nu_lower = -x_sorted[:, ctx.N - 1] - 7.0
        nu_upper = -x_sorted[:, ctx.N] + 7.0

        ls = torch.linspace(0, 1, branch).type_as(x)

        for i in range(ctx.n_iter):
            r = nu_upper - nu_lower
            I = r > ctx.eps  # noqa: E741
            n_update = I.sum()
            if n_update == 0:
                break

            Ix = I.unsqueeze(1).expand_as(x) if old_torch else I

            nus = r[I].unsqueeze(1) * ls + nu_lower[I].unsqueeze(1)
            _xs = x[Ix].view(n_update, 1, nx) + nus.unsqueeze(2)
            fs = torch.sigmoid(_xs).sum(dim=2) - ctx.N
            # assert torch.all(fs[:,0] < 0) and torch.all(fs[:,-1] > 0)

            i_lower = ((fs < 0).sum(dim=1) - 1).long()
            J = i_lower < 0
            if J.sum() > 0:
                print("LML Warning: An example has all positive iterates.")
                i_lower[J] = 0

            i_upper = i_lower + 1

            nu_lower[I] = nus.gather(1, i_lower.unsqueeze(1)).squeeze()
            nu_upper[I] = nus.gather(1, i_upper.unsqueeze(1)).squeeze()

            if J.sum() > 0:
                nu_lower[J] -= 7.0

        if ctx.verbose >= 0 and np.any(I.cpu().numpy()):
            print("LML Warning: Did not converge.")
            # import ipdb; ipdb.set_trace()

        nu = nu_lower + r / 2.0
        y = torch.sigmoid(x + nu.unsqueeze(1))
        if single:
            y = y.squeeze(0)

        if old_torch:
            # Storing these in the object may cause memory leaks.
            ctx.save_for_backward(orig_x)
            ctx.y = y
            ctx.nu = nu
        else:
            ctx.save_for_backward(orig_x, y, nu)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if old_torch:
            (x,) = ctx.saved_tensors
            y = ctx.y
            nu = ctx.nu
        else:
            x, y, nu = ctx.saved_tensors

        single = x.ndimension() == 1
        if single:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)

        assert x.ndimension() == 2
        assert y.ndimension() == 2
        assert grad_output.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= ctx.N:
            dx = torch.zeros_like(x)
            if single:
                dx = dx.squeeze()
            grads = tuple([dx] + [None] * 5)
            return grads

        Hinv = 1.0 / (1.0 / y + 1.0 / (1.0 - y))
        dnu = bdot(Hinv, grad_output) / Hinv.sum(dim=1)
        dx = -Hinv * (-grad_output + dnu.unsqueeze(1))

        if single:
            dx = dx.squeeze()

        grads = tuple([dx] + [None] * 5)
        return grads


if __name__ == "__main__":
    import sys
    from IPython.core import ultratb

    sys.excepthook = ultratb.FormattedTB(
        mode="Verbose", color_scheme="Linux", call_pdb=1
    )

    m = 10
    n = 2

    npr.seed(0)
    x = npr.random(m)

    import cvxpy as cp
    import numdifftools as nd

    y = cp.Variable(m)
    obj = cp.Minimize(-x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y)))
    cons = [0 <= y, y <= 1, cp.sum(y) == n]
    prob = cp.Problem(obj, cons)
    prob.solve(cp.SCS, verbose=True)
    assert "optimal" in prob.status
    y_cp = y.value

    x = Variable(torch.from_numpy(x), requires_grad=True)
    x = torch.stack([x, x])
    y = LML(N=n)(x)

    np.testing.assert_almost_equal(y[0].data.numpy(), y_cp, decimal=3)

    (dy0,) = grad(y[0, 0], x)
    dy0 = dy0.squeeze()

    def f(x):
        x = Variable(torch.from_numpy(x).clone())
        y = LML(N=n)(x)
        return y.data.numpy()

    x = x.data[0].numpy().copy()
    df = nd.Jacobian(f)

    dy0_fd = df(x)[0]

    np.testing.assert_almost_equal(dy0[0].data.numpy(), dy0_fd, decimal=3)
