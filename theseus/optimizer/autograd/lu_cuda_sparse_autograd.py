import torch

from theseus.extlib.cusolver_lu_solver import CusolverLUSolver
from theseus.extlib.mat_mult import mat_vec, mult_MtM, tmat_vec

from ..linear_system import SparseStructure

# if torch.cuda.is_available():


class LUCudaSolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        A_val: torch.Tensor = args[0]
        b: torch.Tensor = args[1]
        sparse_structure: SparseStructure = args[2]
        A_rowPtr: torch.Tensor = args[3]
        A_colInd: torch.Tensor = args[4]
        solver_context: CusolverLUSolver = args[5]
        check_factor_id: bool = args[6]
        # damping: float = args[7]

        AtA_rowPtr = solver_context.A_rowPtr
        AtA_colInd = solver_context.A_colInd

        batch_size = A_val.shape[0]

        AtA = mult_MtM(batch_size, A_rowPtr, A_colInd, A_val, AtA_rowPtr, AtA_colInd)
        solver_context.factor(AtA)

        Atb = tmat_vec(
            batch_size, sparse_structure.num_cols, A_rowPtr, A_colInd, A_val, b
        )
        x = Atb.clone()
        solver_context.solve(x)  # solve in place

        ctx.b = b
        ctx.x = x
        ctx.A_val = A_val
        ctx.A_rowPtr = A_rowPtr
        ctx.A_colInd = A_colInd
        ctx.sparse_structure = sparse_structure
        ctx.solver_context = solver_context

        # HACK: allows to check if the context has been reused (and overwritten)
        ctx.factor_id = solver_context.factor_id if check_factor_id else None

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

    # NOTE: in the torch docs the backward is also marked as "staticmethod", I think it makes sense
    @staticmethod
    def backward(ctx, grad_output):

        # HACK: check if the context has been reused (and overwritten)
        if ctx.factor_id is not None and ctx.factor_id != ctx.solver_context.factor_id:
            raise RuntimeError(
                "Factoring context was overwritten! Increase the number of contexts"
            )

        batch_size = grad_output.shape[0]
        targs = {"dtype": grad_output.dtype, "device": "cuda"}  # grad_output.device}

        H = grad_output.clone()
        ctx.solver_context.solve(H)  # solve in place
        AH = mat_vec(
            batch_size,
            ctx.sparse_structure.num_cols,
            ctx.A_rowPtr,
            ctx.A_colInd,
            ctx.A_val,
            H,
        )
        b_Ax = ctx.b - mat_vec(
            batch_size,
            ctx.sparse_structure.num_cols,
            ctx.A_rowPtr,
            ctx.A_colInd,
            ctx.A_val,
            ctx.x,
        )

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

        return A_grad, AH, None, None, None, None, None
