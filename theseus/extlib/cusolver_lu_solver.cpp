// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "cusolver_sp_defs.h"
#include <pybind11/pybind11.h>
#include <iostream>
#include <torch/extension.h>
#include <functional>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>

enum Ordering {
    AMD = 0,
    RCM,
    MDQ
};

struct CusolverLUSolver {

	~CusolverLUSolver();

    CusolverLUSolver(int batchSize,
                     int64_t numCols,
                     const torch::Tensor& A_rowPtr,
                     const torch::Tensor& A_colInd,
                     Ordering ordering = AMD);

    // returns position of singularity, for each batch element (-1 = no singularity)
    std::vector<int> factor(const torch::Tensor& A_val);

    void solve(const torch::Tensor& b);

    int batchSize;
    int factoredBatchSize;
    int64_t numCols;
    int64_t numRows;
    int64_t nnz;

    torch::Tensor A_rowPtr;
    torch::Tensor A_colInd;
    torch::Tensor P;
    torch::Tensor Q;
    cusolverRfHandle_t cusolverRfH = nullptr;

	// stores the id of the factor stored (to enable workaround related to reusing contexts...)
	int64_t factorId = 0;
};

CusolverLUSolver::~CusolverLUSolver() {
    CUSOLVER_CHECK(cusolverRfDestroy(cusolverRfH));
}

CusolverLUSolver::CusolverLUSolver(int batchSize,
                                   int64_t numCols,
                                   const torch::Tensor& A_rowPtr,
                                   const torch::Tensor& A_colInd,
                                   Ordering ordering)
    : batchSize(batchSize), factoredBatchSize(-1), numCols(numCols), A_rowPtr(A_rowPtr), A_colInd(A_colInd) {

    numRows = A_rowPtr.size(0) - 1;
    nnz = A_colInd.size(0);
    TORCH_CHECK(numRows == numCols); // assume square
    TORCH_CHECK(A_rowPtr.device().is_cuda());
    TORCH_CHECK(A_colInd.device().is_cuda());
    TORCH_CHECK(A_rowPtr.dtype() == torch::kInt);
    TORCH_CHECK(A_colInd.dtype() == torch::kInt);
    TORCH_CHECK(A_rowPtr.dim() == 1);
    TORCH_CHECK(A_colInd.dim() == 1);

    cusolverSpHandle_t cusolverSpH = theseus::cusolver_sp::getCurrentCUDASolverSpHandle();

    cusparseMatDescr_t A_descr = nullptr;
    TORCH_CUDASPARSE_CHECK(cusparseCreateMatDescr(&A_descr));
    TORCH_CUDASPARSE_CHECK(cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO));
    TORCH_CUDASPARSE_CHECK(cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL));

    at::Tensor A_rowPtr_cpu = A_rowPtr.cpu();
    at::Tensor A_colInd_cpu = A_colInd.cpu();
    const int *pA_rowPtr_cpu = A_rowPtr_cpu.data_ptr<int>();
    const int *pA_colInd_cpu = A_colInd_cpu.data_ptr<int>();

    // we compute the permutation Q which allows
    torch::Tensor Qperm = torch::empty(numRows, torch::TensorOptions(torch::kInt));
    int *pQperm = Qperm.data_ptr<int>();

    if (ordering == AMD) {
        CUSOLVER_CHECK(cusolverSpXcsrsymamdHost(cusolverSpH, numRows, nnz,
                                                A_descr, pA_rowPtr_cpu, pA_colInd_cpu,
                                                pQperm));
    } else if (ordering == RCM) {
        CUSOLVER_CHECK(cusolverSpXcsrsymrcmHost(cusolverSpH, numRows, nnz,
                                                A_descr, pA_rowPtr_cpu, pA_colInd_cpu,
                                                pQperm));
    } else if (ordering == MDQ) {
        CUSOLVER_CHECK(cusolverSpXcsrsymmdqHost(cusolverSpH, numRows, nnz,
                                                A_descr, pA_rowPtr_cpu, pA_colInd_cpu,
                                                pQperm));
    } else {
        throw std::runtime_error("CusolverLUSolver: invalid value for ordering: " + std::to_string(ordering));
    }

    // compute the permuted matrix B = Q * A * Qt
    at::Tensor B_rowPtr_cpu = A_rowPtr_cpu.clone();
    at::Tensor B_colInd_cpu = A_colInd_cpu.clone();
    int *pB_rowPtr_cpu = B_rowPtr_cpu.data_ptr<int>();
    int *pB_colInd_cpu = B_colInd_cpu.data_ptr<int>();

    {
        size_t size_perm = 0;
        CUSOLVER_CHECK(cusolverSpXcsrperm_bufferSizeHost(cusolverSpH, numRows, numCols, nnz,
                                                         A_descr, pB_rowPtr_cpu, pB_colInd_cpu,
                                                         pQperm, pQperm, &size_perm));

        torch::Tensor permBuffer = torch::empty(size_perm,
                                                torch::TensorOptions(torch::kByte));
        torch::Tensor permIndices = torch::empty(nnz, // unused
                                                 torch::TensorOptions(torch::kInt));

        CUSOLVER_CHECK(cusolverSpXcsrpermHost(cusolverSpH, numRows, numCols, nnz,
                                              A_descr, pB_rowPtr_cpu, pB_colInd_cpu,
                                              pQperm, pQperm,
                                              permIndices.data_ptr<int>(), permBuffer.data_ptr<uint8_t>()));
    }

    // compute B's factorization with pivoting: B = P*L*Pt * Q*U*Q
    int L_nnz, U_nnz;
    torch::Tensor L_val, L_rowPtr, L_colInd, U_val, U_rowPtr, U_colInd, P_cpu, Q_cpu;

    {
        csrluInfoHost_t info = nullptr;
        CUSOLVER_CHECK(cusolverSpCreateCsrluInfoHost(&info));

        CUSOLVER_CHECK(cusolverSpXcsrluAnalysisHost(cusolverSpH, numRows, nnz,
                                                    A_descr, pB_rowPtr_cpu, pB_colInd_cpu,
                                                    info));

        torch::Tensor B_val_cpu = torch::zeros(nnz, torch::TensorOptions(torch::kDouble));
        double *pB_val_cpu = B_val_cpu.data_ptr<double>();
        // make our model B invertible
        for(int r = 0; r < numRows; r++) {
            // load endpoint `end` at the beginning to avoid recomputation
            for(int i = pB_rowPtr_cpu[r], end = pB_rowPtr_cpu[r+1]; i < end; i++) {
                if(pB_colInd_cpu[i] == r) {
                    pB_val_cpu[i] = 1.0;
                }
            }
        }

        size_t size_internal = 0;
        size_t size_lu  = 0;
        CUSOLVER_CHECK(cusolverSpDcsrluBufferInfoHost(cusolverSpH, numRows, nnz,
                                                      A_descr, pB_val_cpu, pB_rowPtr_cpu, pB_colInd_cpu,
                                                      info,
                                                      &size_internal,
                                                      &size_lu));

        torch::Tensor luBuffer = torch::empty(size_lu, torch::TensorOptions(torch::kByte));
        double pivot_threshold = 1.0;
        double tol = 1e-14;
        CUSOLVER_CHECK(cusolverSpDcsrluFactorHost(cusolverSpH, numRows, nnz,
                                                  A_descr, pB_val_cpu, pB_rowPtr_cpu, pB_colInd_cpu,
                                                  info, pivot_threshold,
                                                  luBuffer.data_ptr<uint8_t>()));

        int singularity = 0;
        CUSOLVER_CHECK(cusolverSpDcsrluZeroPivotHost(cusolverSpH, info, tol, &singularity));
        if (0 <= singularity){
            fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        }

        CUSOLVER_CHECK(cusolverSpXcsrluNnzHost(cusolverSpH, &L_nnz, &U_nnz, info));

        torch::Tensor P_lu = torch::empty(numRows, torch::TensorOptions(torch::kInt));
        torch::Tensor Q_lu = torch::empty(numCols, torch::TensorOptions(torch::kInt));
        L_val = torch::empty(L_nnz, torch::TensorOptions(torch::kDouble));
        L_rowPtr = torch::empty(numRows+1, torch::TensorOptions(torch::kInt));
        L_colInd = torch::empty(L_nnz, torch::TensorOptions(torch::kInt));
        U_val = torch::empty(U_nnz, torch::TensorOptions(torch::kDouble));
        U_rowPtr = torch::empty(numRows+1, torch::TensorOptions(torch::kInt));
        U_colInd = torch::empty(U_nnz, torch::TensorOptions(torch::kInt));

        CUSOLVER_CHECK(cusolverSpDcsrluExtractHost(cusolverSpH,
                                                   P_lu.data_ptr<int>(), Q_lu.data_ptr<int>(),
                                                   A_descr, L_val.data_ptr<double>(), L_rowPtr.data_ptr<int>(), L_colInd.data_ptr<int>(),
                                                   A_descr, U_val.data_ptr<double>(), U_rowPtr.data_ptr<int>(), U_colInd.data_ptr<int>(),
                                                   info,
                                                   luBuffer.data_ptr<uint8_t>()));

        // P, Q (for A's factorization) are obtained as composition of permutations
        P_cpu = torch::empty(numRows, torch::TensorOptions(torch::kInt));
        Q_cpu = torch::empty(numCols, torch::TensorOptions(torch::kInt));
        int* pP = P_cpu.data_ptr<int>(), *pP_lu = P_lu.data_ptr<int>();
        int* pQ = Q_cpu.data_ptr<int>(), *pQ_lu = Q_lu.data_ptr<int>();
        for(int j = 0; j < numRows; j++){
            pP[j] = pQperm[pP_lu[j]];
        }
        for(int j = 0; j < numCols; j++){
            pQ[j] = pQperm[pQ_lu[j]];
        }

        CUSOLVER_CHECK(cusolverSpDestroyCsrluInfoHost(info));
        TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(A_descr));
    }

    // cusolverRf part
    const cusolverRfFactorization_t fact_alg = CUSOLVERRF_FACTORIZATION_ALG0; // default
    const cusolverRfTriangularSolve_t solve_alg = CUSOLVERRF_TRIANGULAR_SOLVE_ALG1; // default
    double nzero = 0.0;
    double nboost = 0.0;
    CUSOLVER_CHECK(cusolverRfCreate(&cusolverRfH));
    CUSOLVER_CHECK(cusolverRfSetNumericProperties(cusolverRfH, nzero, nboost));
    CUSOLVER_CHECK(cusolverRfSetAlgs(cusolverRfH, fact_alg, solve_alg));
    CUSOLVER_CHECK(cusolverRfSetMatrixFormat(cusolverRfH, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));
    CUSOLVER_CHECK(cusolverRfSetResetValuesFastMode(cusolverRfH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));

    at::Tensor A_val_cpu = torch::empty(batchSize * nnz, torch::TensorOptions(torch::kDouble));
    at::Tensor A_val_array_cpu = torch::empty(batchSize * sizeof(double*), torch::TensorOptions(torch::kByte));
    double* pA_val_cpu = A_val_cpu.data_ptr<double>();
    double** pA_val_array_cpu = (double**)A_val_array_cpu.data_ptr<uint8_t>();
    for(int i = 0; i < batchSize; i++) {
        pA_val_array_cpu[i] = pA_val_cpu + nnz * i;
    }

    CUSOLVER_CHECK(cusolverRfBatchSetupHost(batchSize,
                                            numRows, nnz,
                                            A_rowPtr_cpu.data_ptr<int>(), A_colInd_cpu.data_ptr<int>(), pA_val_array_cpu,
                                            L_nnz, L_rowPtr.data_ptr<int>(), L_colInd.data_ptr<int>(), L_val.data_ptr<double>(),
                                            U_nnz, U_rowPtr.data_ptr<int>(), U_colInd.data_ptr<int>(), U_val.data_ptr<double>(),
                                            P_cpu.data_ptr<int>(), Q_cpu.data_ptr<int>(),
                                            cusolverRfH));

    CUSOLVER_CHECK(cusolverRfBatchAnalyze(cusolverRfH));

    P = P_cpu.cuda();
    Q = Q_cpu.cuda();
}

std::vector<int> CusolverLUSolver::factor(const torch::Tensor& A_val) {

    TORCH_CHECK(A_val.device().is_cuda());
    TORCH_CHECK(A_val.dim() == 2);

    // we ideally would like to check "<=" and support irregular (smaller)
    // batch sizes, but (disappointingly) cuda fails unless "==" holds
    TORCH_CHECK(A_val.size(0) == batchSize);
    TORCH_CHECK(A_val.size(1) == nnz);

    factorId++;
    factoredBatchSize = A_val.size(0);

    at::Tensor A_val_array_cpu = torch::empty(factoredBatchSize * sizeof(double*), torch::TensorOptions(torch::kByte));
    double* pA_val = A_val.data_ptr<double>();
    double** pA_val_array_cpu = (double**)A_val_array_cpu.data_ptr<uint8_t>();
    for(int i = 0; i < factoredBatchSize; i++) {
        pA_val_array_cpu[i] = pA_val + nnz * i;
    }
    at::Tensor A_val_array = A_val_array_cpu.cuda();

    CUSOLVER_CHECK(cusolverRfBatchResetValues(factoredBatchSize,
                                              numRows, nnz,
                                              A_rowPtr.data_ptr<int>(), A_colInd.data_ptr<int>(), (double**)A_val_array.data_ptr<uint8_t>(),
                                              P.data_ptr<int>(), Q.data_ptr<int>(),
                                              cusolverRfH));

    CUSOLVER_CHECK(cusolverRfBatchRefactor(cusolverRfH));

    std::vector<int> singularityPositions(factoredBatchSize);
    CUSOLVER_CHECK(cusolverRfBatchZeroPivot(cusolverRfH, singularityPositions.data()));
    for(int i = 0; i < factoredBatchSize; i++) {
        if (singularityPositions[i] >= 0){
            fprintf(stderr, "Error: A[%d] is not invertible, singularity=%d\n", i, singularityPositions[i]);
        }
    }

    return singularityPositions;
}

void CusolverLUSolver::solve(const torch::Tensor& b) {

    TORCH_CHECK(b.device().is_cuda());
    TORCH_CHECK(b.dim() == 2);
    TORCH_CHECK(b.size(0) == factoredBatchSize);
    TORCH_CHECK(b.size(1) == numRows);

    at::Tensor b_array_cpu = torch::empty(factoredBatchSize * sizeof(double*),
                                          torch::TensorOptions(torch::kByte));
    double* pB = b.data_ptr<double>();
    double** pB_array_cpu = (double**)b_array_cpu.data_ptr<uint8_t>();
    for(int i = 0; i < factoredBatchSize; i++) {
        pB_array_cpu[i] = pB + numRows * i;
    }
    at::Tensor b_array = b_array_cpu.cuda();
    at::Tensor temp = torch::empty(numRows * 2 * factoredBatchSize,
                                   torch::TensorOptions(torch::kDouble).device(A_rowPtr.device()));

    CUSOLVER_CHECK(cusolverRfBatchSolve(cusolverRfH,
                                        P.data_ptr<int>(), Q.data_ptr<int>(),
                                        1, // nrhs
                                        temp.data_ptr<double>(), numRows,
                                        (double**)b_array.data_ptr<uint8_t>(), numRows));
}

PYBIND11_MODULE(cusolver_lu_solver, m) {
    m.doc() = "Python bindings for cusolver-based LU solver";
    py::enum_<Ordering>(m, "Ordering",
        "Enumerated class for fill-reducing re-ordering types"
        )
        .value("AMD", AMD, "(Symmetric) Approximate Minimum Degree algorithm based on Quotient Graph")
        .value("RCM", RCM, "(Symmetric) Reverse Cuthill-McKee permutation")
        .value("MDQ", MDQ, "(Symmetric) Minimum Degree algorithm based on Quotient Graph");
    py::class_<CusolverLUSolver>(m, "CusolverLUSolver",
        "Solver class for LU decomposition"
        )
        .def(py::init<int, int64_t, const torch::Tensor&, const torch::Tensor&, Ordering>(),
             "Initialization, it computes the fill-reducing permutation,\n"
             "performs the symbolic factorization, preparing the data structures",
             py::arg("batch_size"),
             py::arg("num_cols"),
             py::arg("A_rowPtr"),
             py::arg("A_colInd"),
             py::arg("ordering") = AMD
        )
        .def("factor", &CusolverLUSolver::factor,
             "Compute the LU factorization, batched. Result be used for one or more 'solve'",
             py::arg("A_val")
        )
        .def("solve", &CusolverLUSolver::solve,
             "Solve in place (b is modified), batch size must match previous call to 'factor'",
             py::arg("b")
        )
	    .def_readonly("factor_id", &CusolverLUSolver::factorId)
	    .def_readonly("batch_size", &CusolverLUSolver::batchSize)
	    .def_readonly("num_rows", &CusolverLUSolver::numRows)
	    .def_readonly("num_cols", &CusolverLUSolver::numCols)
	    .def_readonly("nnz", &CusolverLUSolver::nnz)
	    .def_readonly("A_rowPtr", &CusolverLUSolver::A_rowPtr)
	    .def_readonly("A_colInd", &CusolverLUSolver::A_colInd);
};
