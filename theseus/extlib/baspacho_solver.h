// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/extension.h>

#include "baspacho/baspacho/Solver.h"

struct SymbolicDecompositionData {
    BaSpaCho::SolverPtr solver;

#ifdef THESEUS_HAVE_CUDA
    bool isCuda;
#endif
    torch::Tensor toParamIndex;
    torch::Tensor paramStart;
    torch::Tensor paramSize;
};

// Numeric decomposition, CPU version
class NumericDecomposition {
   public:
    NumericDecomposition(std::shared_ptr<SymbolicDecompositionData> dec,
                         int64_t batchSize);

    void init_factor_data(int64_t batchSize);

    void add_M(const torch::Tensor& val, const torch::Tensor& ptrs,
               const torch::Tensor& inds);

    void add_MtM(const torch::Tensor& val, const torch::Tensor& ptrs,
                 const torch::Tensor& inds);

    void damp(double alpha, double beta);

    void factor();

    void solve(torch::Tensor& x);

#ifdef THESEUS_HAVE_CUDA
    void init_factor_data_cuda(int64_t batchSize);

    void add_M_cuda(const torch::Tensor& val, const torch::Tensor& ptrs,
                    const torch::Tensor& inds);

    void add_MtM_cuda(const torch::Tensor& val, const torch::Tensor& ptrs,
                      const torch::Tensor& inds);

    void damp_cuda(double alpha, double beta);

    void factor_cuda();

    void solve_cuda(torch::Tensor& x);
#endif  // THESEUS_HAVE_CUDA

    std::shared_ptr<SymbolicDecompositionData> dec;
    torch::Tensor data;
};

class SymbolicDecomposition {
   public:
    SymbolicDecomposition(const torch::Tensor& paramSize,
                          const torch::Tensor& sparseStructPtrs,
                          const torch::Tensor& sparseStructInds,
                          const std::string& device);

    NumericDecomposition createNumericDecomposition(int64_t batchSize);

    std::shared_ptr<SymbolicDecompositionData> dec;
};
