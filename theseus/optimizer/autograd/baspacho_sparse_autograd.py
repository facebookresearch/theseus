# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Optional
import torch

from ..linear_system import SparseStructure
from theseus.utils.sparse_matrix_utils import mat_vec, tmat_vec


class BaspachoSolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        from theseus.extlib.baspacho_solver import SymbolicDecomposition

        A_val: torch.Tensor = args[0]
        b: torch.Tensor = args[1]
        sparse_structure: SparseStructure = args[2]
        A_row_ptr: torch.Tensor = args[3]
        A_col_ind: torch.Tensor = args[4]
        symbolic_decomposition: SymbolicDecomposition = args[5]
        damping_alpha_beta: Optional[Tuple[torch.Tensor, torch.Tensor]] = args[6]

        batch_size = A_val.shape[0]

        numeric_decomposition = symbolic_decomposition.create_numeric_decomposition(
            batch_size
        )
        numeric_decomposition.add_MtM(A_val, A_row_ptr, A_col_ind)
        if damping_alpha_beta is not None:
            numeric_decomposition.damp(*damping_alpha_beta)
        numeric_decomposition.factor()

        A_args = sparse_structure.num_cols, A_row_ptr, A_col_ind, A_val
        Atb = tmat_vec(batch_size, *A_args, b)

        x = Atb.clone()
        numeric_decomposition.solve(x)  # solve in place

        ctx.b = b
        ctx.x = x
        ctx.A_val = A_val
        ctx.A_row_ptr = A_row_ptr
        ctx.A_col_ind = A_col_ind
        ctx.sparse_structure = sparse_structure
        ctx.numeric_decomposition = numeric_decomposition
        ctx.damping_alpha_beta = damping_alpha_beta

        return x

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

    # With (large) multiplicative damping, as above with extra terms:
    #      x' = ... - (AtA_damped)^{-1} * alpha*AtA_diag'*x
    # So multiplying by the row vector G we have
    #      G*x' = ... - H * alpha * AtA_diag' * x
    # Note that '...' the part multiplying H[j]*alpha*x[j] is AtA_diag'[j], ie
    # 2 times the scalar product of A's an (A')'s j-th colum. Therefore
    # (A')'s j-th colum is multiplying A's j-th colum by 2*H[j]*alpha*x[j]
    @staticmethod
    def backward(ctx, grad_output):
        batch_size = grad_output.shape[0]
        targs = {"dtype": grad_output.dtype, "device": grad_output.device}

        H = grad_output.clone()
        ctx.numeric_decomposition.solve(H)  # solve in place

        A_args = ctx.sparse_structure.num_cols, ctx.A_row_ptr, ctx.A_col_ind, ctx.A_val
        AH = mat_vec(batch_size, *A_args, H)
        b_Ax = ctx.b - mat_vec(batch_size, *A_args, ctx.x)

        # now we fill values of a matrix with structure identical to A with
        # selected entries from the difference of tensor products:
        #   b_Ax (X) H - AH (X) x
        # NOTE: this row-wise manipulation can be much faster in C++ or Cython
        A_col_ind = ctx.sparse_structure.col_ind
        A_row_ptr = ctx.sparse_structure.row_ptr
        batch_size = grad_output.shape[0]
        A_grad = torch.empty(
            size=(batch_size, len(A_col_ind)), **targs
        )  # return value, A's grad
        for r in range(len(A_row_ptr) - 1):
            start, end = A_row_ptr[r], A_row_ptr[r + 1]
            columns = A_col_ind[start:end]  # col indices, for this row
            A_grad[:, start:end] = (
                b_Ax[:, r].unsqueeze(1) * H[:, columns]
                - AH[:, r].unsqueeze(1) * ctx.x[:, columns]
            )

        # apply correction if there is a multiplicative damping
        if (
            ctx.damping_alpha_beta is not None
            and (ctx.damping_alpha_beta[0] > 0.0).any()
        ):
            alpha = ctx.damping_alpha_beta[0].view(-1, 1)
            alpha2Hx = (alpha * 2.0) * H * ctx.x  # componentwise product
            A_grad -= ctx.A_val * alpha2Hx[:, ctx.A_col_ind.type(torch.long)]

        return A_grad, AH, None, None, None, None, None, None
