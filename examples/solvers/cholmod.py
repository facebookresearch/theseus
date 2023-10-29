from os.path import expanduser

import numpy as np
import scipy
import torch
from sksparse import cholmod

from theseus.utils import Timer

matrix = "raefsky4"  # "cfd2"

A_np_coo = scipy.io.mmread(expanduser(f"~/git/sparse/{matrix}/{matrix}.mtx"))
A_np_csc = scipy.sparse.csc_matrix(A_np_coo)
num_cols = A_np_csc.shape[1]
assert A_np_csc.shape[1] == A_np_csc.shape[0]

np.random.seed(42)
b_np = np.random.randn(num_cols)

t1, t2, t3 = Timer("cpu"), Timer("cpu"), Timer("cpu")
with t1:
    symbolic_decomposition = cholmod.analyze(A_np_csc, mode="supernodal")
with t2:
    factor = symbolic_decomposition.cholesky(A_np_csc)
with t3:
    x_np = factor(b_np)
print(
    f"Setup: {t1.elapsed_time:.03f}s, factor: {t2.elapsed_time:.03f}s, solve: {t3.elapsed_time:.03f}s"
)

b_sol = A_np_csc @ x_np
print("Residue:", np.linalg.norm(b_np - b_sol) / np.linalg.norm(b_np))
