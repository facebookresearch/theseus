# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Tuple

import torch
from sksparse.cholmod import Factor as CholeskyDecomposition

from .common import compute_A_grad
from ..linear_system import SparseStructure

_CholmodSolveFunctionBwdReturnType = Tuple[
    torch.Tensor, torch.Tensor, None, None, None, None
]


class CholmodSolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        At_val: torch.Tensor,
        b: torch.Tensor,
        sparse_structure: SparseStructure,
        symbolic_decomposition: CholeskyDecomposition,
        damping: float,
        detach_hessian: bool = False,
    ) -> torch.Tensor:
        At_val_cpu = At_val.cpu().double()
        b_cpu = b.cpu().double()
        batch_size = At_val.shape[0]
        x_cpu = torch.empty(
            size=(batch_size, sparse_structure.num_cols),
            dtype=At_val.dtype,
            device="cpu",
        )
        cholesky_decompositions = []

        for i in range(batch_size):
            # compute decomposition from symbolic decomposition
            At_i = sparse_structure.csc_transpose(At_val_cpu[i, :])
            cholesky_decomposition = symbolic_decomposition.cholesky_AAt(At_i, damping)

            # solve
            Atb_i = At_i @ b_cpu[i, :]
            x_cpu[i, :] = torch.Tensor(cholesky_decomposition(Atb_i))
            cholesky_decompositions.append(cholesky_decomposition)

        ctx.b_cpu = b_cpu
        ctx.x_cpu = x_cpu
        ctx.At_val_cpu = At_val_cpu
        ctx.sparse_structure = sparse_structure
        ctx.cholesky_decompositions = cholesky_decompositions
        ctx.detach_hessian = detach_hessian

        return x_cpu.to(device=At_val.device, dtype=At_val.dtype)

    # Let v row vector, and w column vector of dimension n, m, and
    # A an nxm matrix. Then
    #   v * A * w = Sum(v_i * A_ij * w_j) = [v (X) w] . A
    # Where by [v (X) w] we mean the nxm matrix which is the
    # tensor product of v and w, and "." is the componentwise
    # dot product of the two nxm matrices.
    #
    # Now, we have
    #      At * A * x = At * b
    #
    # Therefore if A, b, x are parametrized (eg. by u) we have deriving
    #
    # (i)  At'*A*x + At*A'*x + At*A*x' = At'*b + At*b'
    #
    # indicating A'=dA/du, b'=db/du, x'=dx/du
    #
    # Now, assume we have a function f of x, and G = df/dx be the
    # gradient that we consider a row vector, so G*x' is df/du.
    #
    # To compute df/db and df/dA, make x' explicit in the (i):
    #
    #      x' = (At * A)^{-1} (At*b' + At'*b - At'*A*x - At*A'*x)
    #
    # So multiplying by the row vector G we have
    #
    #   G*x' = G*(At*A)^{-1}*At * b' + G*(At*A)^{-1} * (At'*b - At'*A*x - At*A'*x)
    #        = H * At * b' + H * At' * (b - A*x) - H * At * A' * x
    #        = (H*At) * b' + [H (X) (b-A*x)] . At' - [H*At (X) x] . A'
    #        = (H*At) * b' + [(b-A*x) (X) H - H*At (X) x] . A'
    # after putting H = G*(At*A)^{-1} for convenience.
    #
    # Therefore after switching to column vectors we have
    #   df/db = A*H
    # (where H = (At*A)^{-1}*G), while
    #   df/dA = (b-A*x) (X) H - A*H (X) x
    # The two tensor products means that to compute the gradient of
    # a block of A we have to multiply entries taken from (b-A*x) and H,
    # and blocks taken from A*H and x.
    #
    # Here we assume we are provided x and H after the linear solver
    # has been applied to Atb and the gradient G.

    # NOTE: in the torch docs the backward is also marked as "staticmethod", I think it makes sense
    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> _CholmodSolveFunctionBwdReturnType:
        batch_size = grad_output.shape[0]
        H = torch.empty(
            size=(batch_size, ctx.sparse_structure.num_cols), dtype=grad_output.dtype
        )
        AH = torch.empty(
            size=(batch_size, ctx.sparse_structure.num_rows), dtype=grad_output.dtype
        )
        b_Ax = ctx.b_cpu.clone()
        grad_output_cpu = grad_output.cpu()

        for i in range(batch_size):
            H[i, :] = torch.Tensor(
                ctx.cholesky_decompositions[i](grad_output_cpu[i, :])
            )

            A_i = ctx.sparse_structure.csr_straight(ctx.At_val_cpu[i, :])
            AH[i, :] = torch.Tensor(A_i @ H[i, :])
            b_Ax[i, :] -= torch.Tensor(A_i @ ctx.x_cpu[i, :])

        # now we fill values of a matrix with structure identical to A with
        # selected entries from the difference of tensor products:
        #   b_Ax (X) H - AH (X) x
        # NOTE: this row-wise manipulation can be much faster in C++ or Cython
        A_col_ind = ctx.sparse_structure.col_ind
        A_row_ptr = ctx.sparse_structure.row_ptr
        batch_size = grad_output.shape[0]

        A_grad = compute_A_grad(
            batch_size,
            A_row_ptr,
            A_col_ind,
            ctx.b_cpu,
            ctx.x_cpu,
            b_Ax,
            H,
            AH,
            None,
            None,
            None,
            ctx.detach_hessian,
        )
        dev = grad_output.device
        return A_grad.to(device=dev), AH.to(device=dev), None, None, None, None
