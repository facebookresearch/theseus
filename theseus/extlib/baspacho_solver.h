// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/extension.h>

#include "baspacho/baspacho/Solver.h"


// Data stored in a symbolic decomposition, result from analyzing sparse pattern.
//
// Both Python's symbolic decomposition and numeric decomposition will hold a
// shared pointer to the this class
struct SymbolicDecompositionData {
    BaSpaCho::SolverPtr solver;

#ifdef THESEUS_HAVE_CUDA
    bool isCuda; // if cuda is enabled, this flag indicates if this is a cuda context
#endif
    torch::Tensor toParamIndex; // col/row order to param block index
    torch::Tensor paramStart;   // param block index to start
    torch::Tensor paramSize;    // param block index to its size
};

// Numeric decomposition, ie factor data, for a prescribed batch size.
class NumericDecomposition {
   public:
    NumericDecomposition(std::shared_ptr<SymbolicDecompositionData> dec,
                         int64_t batchSize);

    // adds csr matrix M to the factor, batch size must match (auto Cpu/Cuda)
    void add_M(const torch::Tensor& val, const torch::Tensor& ptrs,
               const torch::Tensor& inds);

    // for csr matrix M, adds Mt*M to the factor, batch size must match (auto Cpu/Cuda)
    void add_MtM(const torch::Tensor& val, const torch::Tensor& ptrs,
                 const torch::Tensor& inds);

    // apply damping, ie on factor diag x -> x*(1+alpha) + beta (auto Cpu/Cuda)
    void damp(const torch::Tensor& alpha, const torch::Tensor& beta);

    // in-place solves on vector data, batch size must match (auto Cpu/Cuda)
    void factor();

    // in-place solves on vector data, batch size must match (auto Cpu/Cuda)
    void solve(torch::Tensor& x);

#ifdef THESEUS_HAVE_CUDA
    // adds csr matrix M to the factor, batch size must match (Cuda)
    void add_M_cuda(const torch::Tensor& val, const torch::Tensor& ptrs,
                    const torch::Tensor& inds);

    // for csr matrix M, adds Mt*M to the factor, batch size must match (Cuda)
    void add_MtM_cuda(const torch::Tensor& val, const torch::Tensor& ptrs,
                      const torch::Tensor& inds);

    // apply damping, ie on factor diag x -> x*(1+alpha) + beta (Cuda)
    void damp_cuda(double* alpha, double* beta);

    // computes Cholesky factorization, in-place on the factor (Cuda)
    void factor_cuda();

    // in-place solves on vector data, batch size must match (Cuda)
    void solve_cuda(torch::Tensor& x);
#endif  // THESEUS_HAVE_CUDA

  private:
    // helper, init for a possibly different batch size (Cpu)
    void init_factor_data(int64_t batchSize);

#ifdef THESEUS_HAVE_CUDA
    // helper, init for a possibly different batch size (Cuda)
    void init_factor_data_cuda(int64_t batchSize);
#endif

    std::shared_ptr<SymbolicDecompositionData> dec;
    torch::Tensor data; // factor data
};

// Data stored in a symbolic decomposition, result from analyzing sparse pattern.
// 
// Data are stored with a shared pointer, in this way the Python object can be
// destroyed, and numerical decomposition built from it can still hold a reference
// to the symbolic decomposition data.
class SymbolicDecomposition {
  public:
    SymbolicDecomposition(const torch::Tensor& paramSize,
                          const torch::Tensor& sparseStructPtrs,
                          const torch::Tensor& sparseStructInds,
                          const std::string& device);

    // creates a numeric decomposition, for a prescribed batch size
    NumericDecomposition createNumericDecomposition(int64_t batchSize);

  private:
    std::shared_ptr<SymbolicDecompositionData> dec;
};
