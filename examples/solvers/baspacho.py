from os.path import expanduser

import numpy as np
import scipy
import torch

from theseus.extlib.baspacho_solver import SymbolicDecomposition
from theseus.utils import Timer

matrix = "raefsky4"  # "cfd2"

A_np_coo = scipy.io.mmread(expanduser(f"~/git/sparse/{matrix}/{matrix}.mtx"))
A_np_csr = scipy.sparse.csr_matrix(A_np_coo)
num_cols = A_np_csr.shape[1]
assert A_np_csr.shape[1] == A_np_csr.shape[0]

np.random.seed(42)
b_np = np.random.randn(num_cols)

batch_size = 1
param_size = torch.ones(num_cols, dtype=torch.int64)
A_row_ptr_cpu = torch.tensor(A_np_csr.indptr).long()
A_col_ind_cpu = torch.tensor(A_np_csr.indices).long()

for device in ["cpu", "cuda"]:
    print(f"Device: {device}")
    A_val = torch.tensor(A_np_csr.data).repeat(batch_size, 1).to(device)
    A_row_ptr = A_row_ptr_cpu.to(device)
    A_col_ind = A_col_ind_cpu.to(device)

    b = torch.tensor(b_np).repeat(batch_size, 1).to(device)
    x = b.clone()
    t1, t2, t3 = Timer(device), Timer(device), Timer(device)
    with t1:
        symbolic_decomposition = SymbolicDecomposition(
            param_size, A_row_ptr_cpu, A_col_ind_cpu, device
        )
        numeric_decomposition = symbolic_decomposition.create_numeric_decomposition(
            batch_size
        )
        numeric_decomposition.add_M(A_val, A_row_ptr, A_col_ind)
    with t2:
        numeric_decomposition.factor()
    with t3:
        numeric_decomposition.solve(x)  # solve in place
    print(
        f"Setup: {t1.elapsed_time:.03f}s, factor: {t2.elapsed_time:.03f}s, solve: {t3.elapsed_time:.03f}s"
    )

    b_sol = A_np_csr @ x[0].cpu().numpy()
    print("Residue:", np.linalg.norm(b_np - b_sol) / np.linalg.norm(b_np))
