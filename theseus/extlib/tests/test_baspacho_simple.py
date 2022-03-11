
from scipy.sparse import csr_matrix, tril
from theseus.extlib.baspacho_solver import SymbolicDecomposition
import torch
import numpy as np

def test_simple(verbose=False):
   param_sizes = torch.tensor([2, 3, 5, 2], dtype=torch.int64)
   ss_inds = torch.tensor([0, 0, 1, 1, 2, 0, 3], dtype=torch.int64)
   ss_ptrs = torch.tensor([0, 1, 3, 5, 7], dtype=torch.int64)

   s = SymbolicDecomposition(param_sizes, ss_ptrs, ss_inds)

   batch_size = 2
   f = s.create_numeric_decomposition(batch_size)

   mRowPtr = [
      0, 1, 3, 5, 8, 11, 13, 15, 17, 20, 23, 25, 27
   ]
   mColInd = [
      0, #
      0, 1, #
         1, 2, #
      0,    2, 3, #
         1,    3, 4, #
            2,       5, #
               3,       6, #
                  4,       7, #
            2, 3,             8, #
               3, 4,            9, #
         1,                       10, #
      0,                             11, #
   ]
   mVals = [
   [
      3, #
      .2, 3, #
         -.4,  3, #
      -.7,    -1,  3, #
         .7,       -1, 3, #
            .5,          3, #
               -1,          3, #
                  1,         3, #
            -.3, .6,              3, #
               .3, -.2,            3, #
         1,                           3, #
      -.4,                               3, #    
   ],
   [
      5, #
      .4, 5, #
         1,  5, #
      .7,   -1, 5, #
         1,     3, 5, #
            -.8,      5, #
               -1,       5, #
                  1,       5, #
            .7, -1,            5, #
               .3, .4,           5, #
         1,                          5, #
      .8,                               5, #    
   ]
   ]
   ms = [
      csr_matrix((val, mColInd, mRowPtr), (12, 12))
      for val in mVals
   ]
   if verbose:
      print([m.todense() for m in ms])

   mFulls = [tril(m, -1).transpose().tocsr() + m for m in ms]

   f.add_M(torch.tensor(mVals, dtype=torch.double),
         torch.tensor(mRowPtr, dtype=torch.int64),
         torch.tensor(mColInd, dtype=torch.int64))

   f.factor()

   bData = [
      [1,2,3,-2,-1,-3,0,4,-4,1,2,3],
      [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
   ]
   b = torch.tensor(bData, dtype=torch.double)

   x = b.clone()

   f.solve(x)

   if verbose:
      print("b:", b)
      print("x:", x)
   Mx = torch.tensor(np.array([mFulls[i] @ x[i]
                      for i in range(batch_size)]),
                      dtype=torch.double)
   if verbose:
      print("M*x:", Mx)
   
   residuals = b - Mx
   if verbose:
      print("residuals:", residuals)

   assert all(np.linalg.norm(res) < 1e-10 for res in residuals)
