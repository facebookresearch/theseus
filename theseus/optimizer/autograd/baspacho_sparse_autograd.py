# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Tuple, Optional
import torch

from .common import compute_A_grad
from ..linear_system import SparseStructure
from theseus.utils.sparse_matrix_utils import mat_vec, tmat_vec

_BaspachoSolveFunctionBwdReturnType = Tuple[
    torch.Tensor, torch.Tensor, None, None, None, None, None, None, None
]


class BaspachoSolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        A_val: torch.Tensor,
        b: torch.Tensor,
        sparse_structure: SparseStructure,
        A_row_ptr: torch.Tensor,
        A_col_ind: torch.Tensor,
        symbolic_decomposition: Any,  # actually SymbolicDecomposition
        damping_alpha_beta: Optional[Tuple[torch.Tensor, torch.Tensor]],
        detach_hessian: bool = False,
    ) -> torch.Tensor:
        from theseus.extlib.baspacho_solver import SymbolicDecomposition

        assert isinstance(symbolic_decomposition, SymbolicDecomposition)

        batch_size = A_val.shape[0]

        numeric_decomposition = symbolic_decomposition.create_numeric_decomposition(
            batch_size
        )
        A_val_double = A_val.double()
        numeric_decomposition.add_MtM(A_val_double, A_row_ptr, A_col_ind)
        if damping_alpha_beta is not None:
            numeric_decomposition.damp(
                damping_alpha_beta[0].double(), damping_alpha_beta[1].double()
            )
        numeric_decomposition.factor()

        A_args = sparse_structure.num_cols, A_row_ptr, A_col_ind, A_val_double
        Atb = tmat_vec(batch_size, *A_args, b.double())

        x = Atb.clone()
        numeric_decomposition.solve(x)  # solve in place

        ctx.b = b
        ctx.x = x
        ctx.A_val_double = A_val_double
        ctx.A_row_ptr = A_row_ptr
        ctx.A_col_ind = A_col_ind
        ctx.sparse_structure = sparse_structure
        ctx.numeric_decomposition = numeric_decomposition
        ctx.damping_alpha_beta = damping_alpha_beta
        ctx.detach_hessian = detach_hessian

        return x.to(A_val.dtype)

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
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> _BaspachoSolveFunctionBwdReturnType:
        batch_size = grad_output.shape[0]

        H = grad_output.clone()
        H_double = H.double()
        ctx.numeric_decomposition.solve(H_double)  # solve in place

        A_args = (
            ctx.sparse_structure.num_cols,
            ctx.A_row_ptr,
            ctx.A_col_ind,
            ctx.A_val_double,
        )
        AH = mat_vec(batch_size, *A_args, H_double)
        b_Ax = ctx.b - mat_vec(batch_size, *A_args, ctx.x)

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
            ctx.b,
            ctx.x,
            b_Ax,
            H_double,
            AH,
            ctx.damping_alpha_beta,
            ctx.A_val_double,
            ctx.A_col_ind,
            ctx.detach_hessian,
        )

        return (
            A_grad.to(dtype=grad_output.dtype),
            AH.to(dtype=grad_output.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
