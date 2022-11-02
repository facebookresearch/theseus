// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma nv_diag_suppress 20236
#pragma nv_diag_suppress 20012

#include "baspacho_solver.h"
#include "baspacho/baspacho/CudaDefs.h"
#include "utils.h"

void NumericDecomposition::init_factor_data_cuda(int64_t batchSize) {
    auto xOptions = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    data = torch::zeros({(long)batchSize, (long)(dec->solver->dataSize())},
                        xOptions);
}

__global__ void add_M_kernel(BaSpaCho::PermutedCoalescedAccessor accessor,
                             double* pData, int64_t factorBatchStride,
                             const int64_t* pPtrs, const int64_t* pInds,
                             const int64_t* pToParamIndex, const int64_t* pParamStart,
                             const double* pVal, int64_t valBatchStride,
                             int64_t maxR, int batchSize) {
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIndex >= batchSize || r >= maxR) {
        return;
    }

    int64_t rParam = pToParamIndex[r];
    int64_t rStart = pParamStart[rParam];
    int64_t rowInBlock = r - rStart;
    for (int64_t i = pPtrs[r], iEnd = pPtrs[r + 1]; i < iEnd; i++) {
        int64_t c = pInds[i];
        if (c > r) {
            continue;
        }
        int64_t cParam = pToParamIndex[c];
        int64_t cStart = pParamStart[cParam];
        int64_t colInBlock = c - cStart;

        // note if rParam == cParam then flip = false and
        // we write the lower half
        auto offStrideFlip = accessor.blockOffset(rParam, cParam);
        auto off = std::get<0>(offStrideFlip);
        auto stride = std::get<1>(offStrideFlip);
        auto flip = std::get<2>(offStrideFlip);
        int64_t offsetInFactor =
            off + (flip ? stride * colInBlock + rowInBlock
                        : stride * rowInBlock + colInBlock)
                + batchIndex * factorBatchStride;
        int64_t offsetInMVal = i + batchIndex * valBatchStride;

        pData[offsetInFactor] += pVal[offsetInMVal];
    }

}

void NumericDecomposition::add_M_cuda(const torch::Tensor& val,
                                      const torch::Tensor& ptrs,
                                      const torch::Tensor& inds) {
    int64_t batchSize = data.size(0);
    int64_t factorBatchStride = data.size(1);

    THESEUS_TENSOR_CHECK_CUDA(val, 2, batchSize, torch::kFloat64);
    THESEUS_TENSOR_CHECK_CUDA(ptrs, 1, dec->solver->order() + 1, torch::kInt64);
    THESEUS_TENSOR_CHECK_CUDA(inds, 1, val.size(1), torch::kInt64);


    int64_t valBatchStride = val.size(1);

    const double* pVal = val.data_ptr<double>();
    const int64_t* pPtrs = ptrs.data_ptr<int64_t>();
    const int64_t* pInds = inds.data_ptr<int64_t>();
    int64_t nPtrs = ptrs.size(0);

    // not optimized: for each (r,c), find matrix block and add entry
    auto accessor = dec->solver->deviceAccessor();
    double* pData = data.data_ptr<double>();
    const int64_t* pToParamIndex = dec->toParamIndex.data_ptr<int64_t>();
    const int64_t* pParamStart = dec->paramStart.data_ptr<int64_t>();

    int64_t maxQ = nPtrs - 1;
	dim3 wgs(8, 4);
    while(wgs.y / 2 >= batchSize) { wgs.y /= 2; wgs.x *= 2; }
	dim3 numBlocks((maxQ + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);
    add_M_kernel<<<numBlocks, wgs>>>(accessor, pData, factorBatchStride,
                                     pPtrs, pInds,
                                     pToParamIndex, pParamStart,
                                     pVal, valBatchStride,
                                     maxQ, batchSize);
    cuCHECK(cudaDeviceSynchronize());
}

__global__ void add_MtM_kernel(BaSpaCho::PermutedCoalescedAccessor accessor,
                               double* pData, int64_t factorBatchStride,
                               const int64_t* pPtrs, const int64_t* pInds,
                               const int64_t* pToParamIndex, const int64_t* pParamStart,
                               const double* pVal, int64_t valBatchStride,
                               int64_t maxQ, int batchSize) {
    int64_t q = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIndex >= batchSize || q >= maxQ) {
        return;
    }

    for (int64_t j = pPtrs[q], jEnd = pPtrs[q + 1]; j < jEnd; j++) {
        uint64_t r = pInds[j];
        int64_t rParam = pToParamIndex[r];
        int64_t rStart = pParamStart[rParam];
        int64_t rowInBlock = r - rStart;
        for (int64_t i = pPtrs[q]; i <= j; i++) {
            uint64_t c = pInds[i];
            int64_t cParam = pToParamIndex[c];
            int64_t cStart = pParamStart[cParam];
            int64_t colInBlock = c - cStart;

            // if cParam == rParam then flip == false and we write lower
            // half
            auto offStrideFlip = accessor.blockOffset(rParam, cParam);
            auto off = std::get<0>(offStrideFlip);
            auto stride = std::get<1>(offStrideFlip);
            auto flip = std::get<2>(offStrideFlip);
            int64_t offsetInFactor =
                off + (flip ? stride * colInBlock + rowInBlock
                            : stride * rowInBlock + colInBlock)
                + batchIndex * factorBatchStride;
            int64_t offsetInMValR = j + batchIndex * valBatchStride;
            int64_t offsetInMValC = i + batchIndex * valBatchStride;
            atomicAdd(pData + offsetInFactor, pVal[offsetInMValR] * pVal[offsetInMValC]);
        }
    }
}

void NumericDecomposition::add_MtM_cuda(const torch::Tensor& val,
                                        const torch::Tensor& ptrs,
                                        const torch::Tensor& inds) {
    int64_t batchSize = data.size(0);
    int64_t factorBatchStride = data.size(1);

    THESEUS_TENSOR_CHECK_CUDA(val, 2, batchSize, torch::kFloat64);
    THESEUS_TENSOR_CHECK_CUDA(ptrs, 1, ptrs.size(0), torch::kInt64);
    THESEUS_TENSOR_CHECK_CUDA(inds, 1, val.size(1), torch::kInt64);
    
    int64_t valBatchStride = val.size(1);

    const double* pVal = val.data_ptr<double>();
    const int64_t* pPtrs = ptrs.data_ptr<int64_t>();
    const int64_t* pInds = inds.data_ptr<int64_t>();
    int64_t nPtrs = ptrs.size(0);

    // not optimized: for each (r,c), find matrix block and add entry
    auto accessor = dec->solver->deviceAccessor();
    double* pData = data.data_ptr<double>();
    const int64_t* pToParamIndex = dec->toParamIndex.data_ptr<int64_t>();
    const int64_t* pParamStart = dec->paramStart.data_ptr<int64_t>();

    int64_t maxQ = nPtrs - 1;
	dim3 wgs(8, 4);
    while(wgs.y / 2 >= batchSize) { wgs.y /= 2; wgs.x *= 2; }
	dim3 numBlocks((maxQ + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);
    add_MtM_kernel<<<numBlocks, wgs>>>(accessor, pData, factorBatchStride,
                                       pPtrs, pInds,
                                       pToParamIndex, pParamStart,
                                       pVal, valBatchStride,
                                       maxQ, batchSize);
    cuCHECK(cudaDeviceSynchronize());
}

__global__ void damp_kernel(BaSpaCho::PermutedCoalescedAccessor accessor,
                     double* pFactor, int64_t factorSize,
                     double* alpha, double* beta,
                     int64_t maxI, int batchSize) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIndex >= batchSize || i >= maxI) {
        return;
    }

    double* pFactorItem = pFactor + factorSize * batchIndex;
    auto block = accessor.diagBlock(pFactorItem, i);
    block.diagonal() *= (1.0 + alpha[batchIndex]);
    block.diagonal().array() += beta[batchIndex];
}

void NumericDecomposition::damp_cuda(double* alpha, double* beta) {
    int64_t batchSize = data.size(0);
    int64_t factorSize = data.size(1);
    double* pFactor = data.data_ptr<double>();

    int64_t nParams = dec->paramSize.size(0);
    auto accessor = dec->solver->deviceAccessor();

	dim3 wgs(8, 4);
    while(wgs.y / 2 >= batchSize) { wgs.y /= 2; wgs.x *= 2; }
	dim3 numBlocks((nParams + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);

    damp_kernel<<<numBlocks, wgs>>>(accessor, pFactor, factorSize, alpha, beta, nParams, batchSize);
    cuCHECK(cudaDeviceSynchronize());
}

void NumericDecomposition::factor_cuda() {
    int64_t batchSize = data.size(0);
    int64_t factorSize = data.size(1);
    double* pFactor = data.data_ptr<double>();

    // no proper support for batched factor on CPU, iterate
    std::vector<double*> factorPtrs;
    for (int i = 0; i < batchSize; i++) {
        factorPtrs.push_back(pFactor + factorSize * i);
    }
    dec->solver->factor(&factorPtrs);
}

__global__ void scramble_kernel(BaSpaCho::PermutedCoalescedAccessor acc,
                         const int64_t* pParamSize, const int64_t* pParamStart,
                         const int64_t order,
                         const double* pX, double* pPermX, int64_t maxI,
                         int batchSize) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIndex >= batchSize || i >= maxI) {
        return;
    }

    int64_t size = pParamSize[i];
    int64_t origStart = pParamStart[i] + order * batchIndex;
    int64_t destStart = acc.paramStart(i) + order * batchIndex;
    Eigen::Map<Eigen::VectorXd>(pPermX + destStart, size)
        = Eigen::Map<const Eigen::VectorXd>(pX + origStart, size);
}

__global__ void unscramble_kernel(BaSpaCho::PermutedCoalescedAccessor acc,
                         const int64_t* pParamSize, const int64_t* pParamStart,
                         const int64_t order,
                         double* pX, const double* pPermX, int64_t maxI,
                         int batchSize) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (batchIndex >= batchSize || i >= maxI) {
        return;
    }

    int64_t size = pParamSize[i];
    int64_t origStart = pParamStart[i] + order * batchIndex;
    int64_t destStart = acc.paramStart(i) + order * batchIndex;
    Eigen::Map<Eigen::VectorXd>(pX + origStart, size)
        = Eigen::Map<const Eigen::VectorXd>(pPermX + destStart, size);
}

void NumericDecomposition::solve_cuda(torch::Tensor& x) {
    int64_t batchSize = data.size(0);
    int64_t order = dec->solver->order();
    THESEUS_TENSOR_CHECK_CUDA(x, 2, batchSize, x.dtype());
    TORCH_CHECK(x.size(1) == order);

    using OuterStride = Eigen::OuterStride<>;
    using OuterStridedM = Eigen::Map<Eigen::MatrixXd, 0, OuterStride>;

    // scramble according to parameter permutation
    auto accessor = dec->solver->deviceAccessor();
    auto xOptions = torch::TensorOptions().dtype(torch::kFloat64).device(data.device());;
    torch::Tensor permX =
        torch::empty({(long)(batchSize), (long)(order)}, xOptions);
    const int64_t* pParamSize = dec->paramSize.data_ptr<int64_t>();
    const int64_t* pParamStart = dec->paramStart.data_ptr<int64_t>();
    int64_t nParams = dec->paramSize.size(0);
    double* pX = x.data_ptr<double>();
    double* pPermX = permX.data_ptr<double>();

	dim3 wgs(8, 4);
    while(wgs.y / 2 >= batchSize) { wgs.y /= 2; wgs.x *= 2; }
	dim3 numBlocks((nParams + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);
    scramble_kernel<<<numBlocks, wgs>>>(accessor, pParamSize, pParamStart,
                                 order, pX, pPermX, nParams, batchSize);
    cuCHECK(cudaDeviceSynchronize());

    // no proper support for batched solve on CPU, iterate
    double* pFactor = data.data_ptr<double>();
    int64_t factorSize = data.size(1);
    std::vector<double*> factorPtrs, xPtrs;
    for (int i = 0; i < batchSize; i++) {
        factorPtrs.push_back(pFactor + factorSize * i);
        xPtrs.push_back(pPermX + order * i);
    }
    dec->solver->solve(&factorPtrs, &xPtrs, order, 1);

    // un-scramble
    unscramble_kernel<<<numBlocks, wgs>>>(accessor, pParamSize, pParamStart,
                                   order, pX, pPermX, nParams, batchSize);
    cuCHECK(cudaDeviceSynchronize());
}
