
#include <torch/extension.h>

#include <iostream>

#include "baspacho/baspacho/Solver.h"
#include "baspacho/testing/TestingUtils.h"

using namespace BaSpaCho;
using namespace testing;
using namespace std;

struct SymbolicDecompositionData {
    BaSpaCho::SolverPtr solver;
    torch::Tensor toParamIndex;
    torch::Tensor paramStart;
    torch::Tensor paramSize;
};

class NumericDecomposition {
   public:
    NumericDecomposition(std::shared_ptr<SymbolicDecompositionData> dec,
                         int64_t batchSize);

    void add_M(const torch::Tensor& val, const torch::Tensor& ptrs,
                      const torch::Tensor& inds);

    void add_MtM(const torch::Tensor& val, const torch::Tensor& ptrs,
                 const torch::Tensor& inds);

    void factor();

    void solve(torch::Tensor& x);

    std::shared_ptr<SymbolicDecompositionData> dec;
    torch::Tensor data;
};

NumericDecomposition::NumericDecomposition(std::shared_ptr<SymbolicDecompositionData> dec,
                                           int64_t batchSize)
    : dec(dec) {
    auto xOptions = torch::TensorOptions().dtype(torch::kFloat64);
    data = torch::zeros({(long)batchSize, 
                        (long)(dec->solver->dataSize())}, xOptions);
}

void NumericDecomposition::add_M(const torch::Tensor& val,
                                 const torch::Tensor& ptrs,
                                 const torch::Tensor& inds) {
    int64_t batchSize = data.size(0);
    int64_t factorBatchStride = data.size(1);

    TORCH_CHECK(val.device().is_cpu());
    TORCH_CHECK(ptrs.device().is_cpu());
    TORCH_CHECK(inds.device().is_cpu());
    TORCH_CHECK(val.dtype() == torch::kFloat64);
    TORCH_CHECK(ptrs.dtype() == torch::kInt64);
    TORCH_CHECK(inds.dtype() == torch::kInt64);
    TORCH_CHECK(val.dim() == 2);
    TORCH_CHECK(ptrs.dim() == 1);
    TORCH_CHECK(inds.dim() == 1);
    TORCH_CHECK(val.size(0) == batchSize);
    TORCH_CHECK(val.size(1) == inds.size(0));
    TORCH_CHECK(ptrs.size(0) - 1 == dec->solver->order());

    int64_t valBatchStride = val.size(1);

    const double* pVal = val.data_ptr<double>();
    const int64_t* pPtrs = ptrs.data_ptr<int64_t>();
    const int64_t* pInds = inds.data_ptr<int64_t>();
    TORCH_CHECK(pPtrs[ptrs.size(0) - 1] == inds.size(0));
    int64_t nPtrs = ptrs.size(0);

    // not optimized: for each (r,c), find matrix block and add entry
    auto accessor = dec->solver->accessor();
    double* pData = data.data_ptr<double>();
    const int64_t* pToParamIndex = dec->toParamIndex.data_ptr<int64_t>();
    const int64_t* pParamStart = dec->paramStart.data_ptr<int64_t>();
    for (int64_t r = 0; r < nPtrs - 1; r++) {
        int64_t rParam = pToParamIndex[r];
        int64_t rStart = pParamStart[rParam];
        int64_t rowInBlock = r - rStart;
        for (int64_t i = pPtrs[r], iEnd = pPtrs[r + 1]; i < iEnd; i++) {
            uint64_t c = pInds[i];
            if(c > r) {
                continue;
            }
            int64_t cParam = pToParamIndex[c];
            int64_t cStart = pParamStart[cParam];
            int64_t colInBlock = c - cStart;

            // note if rParam == cParam then flip = false and
            // we write the lower half
            auto [off, stride, flip] = accessor.blockOffset(rParam, cParam);
            int64_t offsetInFactor = off + (flip
                                    ? stride * colInBlock + rowInBlock
                                    : stride * rowInBlock + colInBlock);
            int64_t offsetInMVal = i;

            for(int b = 0; b < batchSize; b++) {
                pData[offsetInFactor] += pVal[offsetInMVal];
                offsetInFactor += factorBatchStride;
                offsetInMVal += valBatchStride;
            }
        }
    }
}

void NumericDecomposition::add_MtM(const torch::Tensor& val,
                                   const torch::Tensor& ptrs,
                                   const torch::Tensor& inds) {
    int64_t batchSize = data.size(0);
    int64_t factorBatchStride = data.size(1);

    TORCH_CHECK(val.device().is_cpu());
    TORCH_CHECK(ptrs.device().is_cpu());
    TORCH_CHECK(inds.device().is_cpu());
    TORCH_CHECK(val.dtype() == torch::kFloat64);
    TORCH_CHECK(ptrs.dtype() == torch::kInt64);
    TORCH_CHECK(inds.dtype() == torch::kInt64);
    TORCH_CHECK(val.dim() == 2);
    TORCH_CHECK(ptrs.dim() == 1);
    TORCH_CHECK(inds.dim() == 1);
    TORCH_CHECK(val.size(0) == batchSize);
    TORCH_CHECK(val.size(1) == inds.size(0));

    int64_t valBatchStride = val.size(1);

    const double* pVal = val.data_ptr<double>();
    const int64_t* pPtrs = ptrs.data_ptr<int64_t>();
    const int64_t* pInds = inds.data_ptr<int64_t>();
    TORCH_CHECK(pPtrs[ptrs.size(0) - 1] == inds.size(0));
    int64_t nPtrs = ptrs.size(0);

    // not optimized: for each (r,c), find matrix block and add entry
    auto accessor = dec->solver->accessor();
    double* pData = data.data_ptr<double>();
    const int64_t* pToParamIndex = dec->toParamIndex.data_ptr<int64_t>();
    const int64_t* pParamStart = dec->paramStart.data_ptr<int64_t>();
    // for each row in M...
    for (int64_t q = 0; q < nPtrs - 1; q++) {
        // iterate on i <= j (corresponding to c <= r via inds)
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

                // if cParam == rParam then flip == false and we write lower half
                auto [off, stride, flip] = accessor.blockOffset(rParam, cParam);
                int64_t offsetInFactor = off + (flip ? stride * colInBlock + rowInBlock : stride * rowInBlock + colInBlock);
                int64_t offsetInMValR = j;
                int64_t offsetInMValC = i;

                for(int b = 0; b < batchSize; b++) {
                    pData[offsetInFactor] += pVal[offsetInMValR] * pVal[offsetInMValC];
                    offsetInFactor += factorBatchStride;
                    offsetInMValR += valBatchStride;
                    offsetInMValC += valBatchStride;
                }
            }
        }
    }
}

void NumericDecomposition::factor() {
    int64_t batchSize = data.size(0);
    int64_t factorSize = data.size(1);
    double* pFactor = data.data_ptr<double>();

    // no proper support for batched factor on CPU, iterate
    for(int i = 0; i < batchSize; i++) {
        dec->solver->factor(pFactor + factorSize * i);
    }
}

void NumericDecomposition::solve(torch::Tensor& x) {
    int64_t batchSize = data.size(0);
    int64_t order = dec->solver->order();
    TORCH_CHECK(x.device().is_cpu());
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(x.size(0) == batchSize);
    TORCH_CHECK(x.size(1) == order);

    using OuterStride = Eigen::OuterStride<>;
    using OuterStridedM = Eigen::Map<Eigen::MatrixXd, 0, OuterStride>;

    // scramble according to parameter permutation
    auto accessor = dec->solver->accessor();
    auto xOptions = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor permX = torch::empty({(long)(batchSize), 
                                (long)(order)}, xOptions);
    const int64_t* pParamSize = dec->paramSize.data_ptr<int64_t>();
    const int64_t* pParamStart = dec->paramStart.data_ptr<int64_t>();
    int64_t nParams = dec->paramSize.size(0);
    double* pX = x.data_ptr<double>();
    double* pPermX = permX.data_ptr<double>();
    for (int64_t i = 0; i < nParams; i++) {
        int64_t size = pParamSize[i];
        int64_t origStart = pParamStart[i];
        int64_t destStart = accessor.paramStart(i);
        OuterStridedM(pPermX + destStart, size, batchSize, OuterStride(order))
            = OuterStridedM(pX + origStart, size, batchSize, OuterStride(order));
    }

    // no proper support for batched solve on CPU, iterate
    double* pFactor = data.data_ptr<double>();
    int64_t factorSize = data.size(1);
    for(int i = 0; i < batchSize; i++) {
        dec->solver->solve(pFactor + factorSize * i, pPermX + order * i, order, 1);
    }

    // un-scramble
    for (int64_t i = 0; i < nParams; i++) {
        int64_t size = pParamSize[i];
        int64_t origStart = pParamStart[i];
        int64_t destStart = accessor.paramStart(i);
        OuterStridedM(pX + origStart, size, batchSize, OuterStride(order)) = 
            OuterStridedM(pPermX + destStart, size, batchSize, OuterStride(order));
    }
}


class SymbolicDecomposition {
   public:
    SymbolicDecomposition(const torch::Tensor& paramSize,
                          const torch::Tensor& sparseStructPtrs,
                          const torch::Tensor& sparseStructInds);

    NumericDecomposition createNumericDecomposition(int64_t batchSize) {
        return NumericDecomposition(dec, batchSize);  // TODO: CUDA/CPU flags
    }

    std::shared_ptr<SymbolicDecompositionData> dec;
};

SymbolicDecomposition::SymbolicDecomposition(const torch::Tensor& paramSize,
                                             const torch::Tensor& sparseStructPtrs,
                                             const torch::Tensor& sparseStructInds) {
    TORCH_CHECK(paramSize.device().is_cpu());
    TORCH_CHECK(sparseStructPtrs.device().is_cpu());
    TORCH_CHECK(sparseStructInds.device().is_cpu());
    TORCH_CHECK(paramSize.dtype() == torch::kInt64);
    TORCH_CHECK(sparseStructPtrs.dtype() == torch::kInt64);
    TORCH_CHECK(sparseStructInds.dtype() == torch::kInt64);
    TORCH_CHECK(paramSize.dim() == 1);
    TORCH_CHECK(sparseStructPtrs.dim() == 1);
    TORCH_CHECK(sparseStructInds.dim() == 1);
    TORCH_CHECK(paramSize.size(0) + 1 == sparseStructPtrs.size(0));

    int64_t nParams = paramSize.size(0);
    int64_t nPtrs = sparseStructPtrs.size(0);
    int64_t nInds = sparseStructInds.size(0);
    const int64_t* pParamSize = paramSize.data_ptr<int64_t>();
    const int64_t* pPtrs = sparseStructPtrs.data_ptr<int64_t>();
    const int64_t* pInds = sparseStructInds.data_ptr<int64_t>();

    vector<int64_t> paramSizeVec(pParamSize, pParamSize + nParams);
    vector<int64_t> ptrsVec(pPtrs, pPtrs + nPtrs);
    vector<int64_t> indsVec(pInds, pInds + nInds);

    SparseStructure ss(move(ptrsVec), move(indsVec));
    dec = std::make_shared<SymbolicDecompositionData>();
    dec->solver = createSolver({}, paramSizeVec, ss);

    // those data will be used
    int64_t totParamSize = 0;
    for (auto p : paramSizeVec) {
        totParamSize += p;
    }
    auto xOptions = torch::TensorOptions().dtype(torch::kInt64);
    dec->toParamIndex = torch::empty({(long)totParamSize}, xOptions);
    dec->paramSize = paramSize;
    dec->paramStart = torch::empty({(long)paramSize.size(0)}, xOptions);
    int64_t* toParamIndexP = dec->toParamIndex.data_ptr<int64_t>();
    int64_t* paramStartP = dec->paramStart.data_ptr<int64_t>();
    for (size_t i = 0, start = 0; i < paramSizeVec.size(); i++) {
        paramStartP[i] = start;
        start += paramSizeVec[i];
        for (int j = 0; j < paramSizeVec[i]; j++) {
            *toParamIndexP++ = i;
        }
    }
}

PYBIND11_MODULE(baspacho_solver, m) {
    m.doc() = "Python bindings for BaSpaCho solver";
    py::class_<SymbolicDecomposition>(m, "SymbolicDecomposition",
                                      "Symbolic decomposition")
        .def(py::init<const torch::Tensor&, const torch::Tensor&,
                      const torch::Tensor&>(),
             "Initialization, it computes the fill-reducing permutation,\n"
             "performs the symbolic factorization, preparing the data\n"
             "structures. It takes as inputs the ptrs/inds of the sparse\n"
             "block-structure, and param_size with sizes of blocks",
             py::arg("param_size"), py::arg("sparse_struct_ptrs"),
             py::arg("sparse_struct_inds"))
        .def("create_numeric_decomposition",
             &SymbolicDecomposition::createNumericDecomposition,
             "Creates an object that can contain a factor, that can "
             "be edited and factored",
             py::arg("batch_size"));
    py::class_<NumericDecomposition>(m, "NumericDecomposition",
                                     "Numeric decomposition")
        .def("add_M", &NumericDecomposition::add_M,
             "Adds a csr Matrix to the factor data", py::arg("val"), py::arg("ptrs"),
             py::arg("inds"))
        .def("add_MtM", &NumericDecomposition::add_MtM,
             "Adds Mt * M to the factor data", py::arg("val"), py::arg("ptrs"),
             py::arg("inds"))
        .def("factor", &NumericDecomposition::factor,
             "Computed the Cholesky decomposition (factor data are replaced)")
        .def("solve", &NumericDecomposition::solve, "Solves (in-place)",
             py::arg("x"));
};
