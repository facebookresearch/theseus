
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

__device__ int bisect_index(const int* values, int len, int needle) {

	int a = 0, b = len;
	while (b > a + 1) {
		int m = (a + b) / 2;
		if(values[m] > needle) {
			b = m;
		} else {
			a = m;
		}
	}
	if(values[a] != needle) {
		printf("Error!! needle %d not found in array of length %d\n", needle, len);
	}
	return a;
}

__global__ void mult_MtM_kernel(int batchSize,
                                int M_rows,
                                int M_nnz,
                                const int* M_rowPtr,
                                const int* M_colInd,
                                const double* Ms_val,
                                int MtM_rows,
                                int MtM_nnz,
                                const int* MtM_rowPtr,
                                const int* MtM_colInd,
                                double* MtMs_val) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if(batchIndex >= batchSize || row >= M_rows) {
		return;
	}

	// matrices are in CSR format:
	//   rowPtr determines begin/end of row data,
	//   colInd determines the column index
	int srcRow_offset = M_rowPtr[row];
	int srcRow_len = M_rowPtr[row+1] - srcRow_offset;
	const int* srcRow_colInd = M_colInd + srcRow_offset;
	const double* srcRow_val = Ms_val + batchIndex * M_nnz + srcRow_offset;
	double* MtMs_batch_val = MtMs_val + batchIndex * MtM_nnz;
	for(int i = 0; i < srcRow_len; i++) {
		int dstRow = srcRow_colInd[i];
		int dstRow_offset = MtM_rowPtr[dstRow];
		int dstRow_len = MtM_rowPtr[dstRow + 1] - MtM_rowPtr[dstRow];
		const int* dstRow_colInd = MtM_colInd + dstRow_offset;
		double* dstRow_val = MtMs_batch_val + dstRow_offset;
		for(int j = 0; j < srcRow_len; j++) {
			double val = srcRow_val[i] * srcRow_val[j];
			int dstCol = srcRow_colInd[j];

			// The result has a different sparsity pattern. Therefore we have to
			// identify where the destination's `colInd` is `dstCol`, working
			// in row of order `dstRow` in destination
			int positionInDstRow = bisect_index(dstRow_colInd, dstRow_len, dstCol);
			atomicAdd(dstRow_val + positionInDstRow, val);
		}
	}
}

torch::Tensor mult_MtM(int batchSize,
                       const torch::Tensor& M_rowPtr,
                       const torch::Tensor& M_colInd,
                       const torch::Tensor& Ms_val,
                       const torch::Tensor& MtM_rowPtr,
                       const torch::Tensor& MtM_colInd) {

	int64_t M_rows = M_rowPtr.size(0) - 1;
	int64_t M_nnz = M_colInd.size(0);

	TORCH_CHECK(M_rowPtr.device().is_cuda());
	TORCH_CHECK(M_colInd.device().is_cuda());
	TORCH_CHECK(Ms_val.device().is_cuda());
	TORCH_CHECK(M_rowPtr.dtype() == torch::kInt);
	TORCH_CHECK(M_colInd.dtype() == torch::kInt);
	TORCH_CHECK(Ms_val.dtype() == torch::kDouble); // TODO: add support for float
	TORCH_CHECK(M_rowPtr.dim() == 1);
	TORCH_CHECK(M_colInd.dim() == 1);
	TORCH_CHECK(Ms_val.dim() == 2);
	TORCH_CHECK(Ms_val.size(0) == batchSize);
	TORCH_CHECK(Ms_val.size(1) == M_nnz);

	int64_t MtM_rows = MtM_rowPtr.size(0) - 1;
	int64_t MtM_nnz = MtM_colInd.size(0);
	
	TORCH_CHECK(MtM_rowPtr.device().is_cuda());
	TORCH_CHECK(MtM_colInd.device().is_cuda());
	TORCH_CHECK(MtM_rowPtr.dim() == 1);
	TORCH_CHECK(MtM_colInd.dim() == 1);

	auto xOptions = torch::TensorOptions().dtype(torch::kDouble).device(Ms_val.device());
	torch::Tensor MtMs_val = torch::zeros({(long)batchSize, (long)MtM_nnz}, xOptions);

	// TODO: do experiments on choice of work group size
	dim3 wgs(1, 16);
	dim3 numBlocks((M_rows + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);

    M_rowPtr.data_ptr<int>();
    M_colInd.data_ptr<int>();
    Ms_val.data_ptr<double>();
    MtM_rowPtr.data_ptr<int>();
    MtM_colInd.data_ptr<int>();
    MtMs_val.data_ptr<double>();

    // TODO: set stream according to torch
    mult_MtM_kernel<<<numBlocks, wgs>>>(batchSize,
                                        M_rows,
                                        M_nnz,
                                        M_rowPtr.data_ptr<int>(),
                                        M_colInd.data_ptr<int>(),
                                        Ms_val.data_ptr<double>(),
                                        MtM_rows,
                                        MtM_nnz,
                                        MtM_rowPtr.data_ptr<int>(),
                                        MtM_colInd.data_ptr<int>(),
                                        MtMs_val.data_ptr<double>());
	return MtMs_val;
}

__global__ void mat_vec_kernel(int batchSize,
                               int M_rows,
                               int M_cols,
                               int M_nnz,
                               const int* M_rowPtr,
                               const int* M_colInd,
                               const double* Ms_val,
                               const double* vec,
                               double* retv) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if(batchIndex >= batchSize || row >= M_rows) {
		return;
	}
	
	int srcRow_offset = M_rowPtr[row];
	int srcRow_len = M_rowPtr[row+1] - srcRow_offset;
	const int* srcRow_colInd = M_colInd + srcRow_offset;
	const double* srcRow_val = Ms_val + batchIndex * M_nnz + srcRow_offset;
	const double* srcVec = vec + batchIndex * M_cols;

	double value = 0.0;
	for(int i = 0; i < srcRow_len; i++) {
		value += srcRow_val[i] * srcVec[srcRow_colInd[i]];
	}

	*(retv + batchIndex * M_rows + row) = value;
}

torch::Tensor mat_vec(int batchSize,
                      int M_cols,
                      const torch::Tensor& M_rowPtr,
                      const torch::Tensor& M_colInd,
                      const torch::Tensor& Ms_val,
                      const torch::Tensor& vec) {

	int64_t M_rows = M_rowPtr.size(0) - 1;
	int64_t M_nnz = M_colInd.size(0);

	TORCH_CHECK(M_rowPtr.device().is_cuda());
	TORCH_CHECK(M_colInd.device().is_cuda());
	TORCH_CHECK(Ms_val.device().is_cuda());
	TORCH_CHECK(M_rowPtr.dtype() == torch::kInt);
	TORCH_CHECK(M_colInd.dtype() == torch::kInt);
	TORCH_CHECK(Ms_val.dtype() == torch::kDouble); // TODO: add support for float
	TORCH_CHECK(M_rowPtr.dim() == 1);
	TORCH_CHECK(M_colInd.dim() == 1);
	TORCH_CHECK(Ms_val.dim() == 2);
	TORCH_CHECK(Ms_val.size(0) == batchSize);
	TORCH_CHECK(Ms_val.size(1) == M_nnz);
	TORCH_CHECK(vec.device().is_cuda());
	TORCH_CHECK(vec.dim() == 2);
	TORCH_CHECK(vec.size(0) == batchSize);
	TORCH_CHECK(vec.size(1) == M_cols);
	
	auto xOptions = torch::TensorOptions().dtype(torch::kDouble).device(Ms_val.device());
	torch::Tensor retv = torch::empty({(long)batchSize, (long)M_rows}, xOptions);

	// TODO: do experiments on choice of work group size
	dim3 wgs(1, 16);
	dim3 numBlocks((M_rows + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);

	mat_vec_kernel<<<numBlocks, wgs>>>(batchSize,
	                                   M_rows,
	                                   M_cols,
	                                   M_nnz,
	                                   M_rowPtr.data_ptr<int>(),
	                                   M_colInd.data_ptr<int>(),
	                                   Ms_val.data_ptr<double>(),
	                                   vec.data_ptr<double>(),
	                                   retv.data_ptr<double>());
	return retv;
}



__global__ void tmat_vec_kernel(int batchSize,
                                int M_rows,
                                int M_cols,
                                int M_nnz,
                                const int* M_rowPtr,
                                const int* M_colInd,
                                const double* Ms_val,
                                const double* vec,
                                double* retv) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if(batchIndex >= batchSize || row >= M_rows) {
		return;
	}
	
	int srcRow_offset = M_rowPtr[row];
	int srcRow_len = M_rowPtr[row+1] - srcRow_offset;
	const int* srcRow_colInd = M_colInd + srcRow_offset;
	const double* srcRow_val = Ms_val + batchIndex * M_nnz + srcRow_offset;
	double vecVal = vec[batchIndex * M_rows + row];
	double* dstVec = retv + batchIndex * M_cols;

	for(int i = 0; i < srcRow_len; i++) {
		atomicAdd(dstVec + srcRow_colInd[i], vecVal * srcRow_val[i]);
	}
}

torch::Tensor tmat_vec(int batchSize,
                       int M_cols,
                       const torch::Tensor& M_rowPtr,
                       const torch::Tensor& M_colInd,
                       const torch::Tensor& Ms_val,
                       const torch::Tensor& vec) {

	int64_t M_rows = M_rowPtr.size(0) - 1;
	int64_t M_nnz = M_colInd.size(0);

	TORCH_CHECK(M_rowPtr.device().is_cuda());
	TORCH_CHECK(M_colInd.device().is_cuda());
	TORCH_CHECK(Ms_val.device().is_cuda());
	TORCH_CHECK(M_rowPtr.dtype() == torch::kInt);
	TORCH_CHECK(M_colInd.dtype() == torch::kInt);
	TORCH_CHECK(Ms_val.dtype() == torch::kDouble); // TODO: add support for float
	TORCH_CHECK(M_rowPtr.dim() == 1);
	TORCH_CHECK(M_colInd.dim() == 1);
	TORCH_CHECK(Ms_val.dim() == 2);
	TORCH_CHECK(Ms_val.size(0) == batchSize);
	TORCH_CHECK(Ms_val.size(1) == M_nnz);
	TORCH_CHECK(vec.device().is_cuda());
	TORCH_CHECK(vec.dim() == 2);
	TORCH_CHECK(vec.size(0) == batchSize);
	TORCH_CHECK(vec.size(1) == M_rows);
	
	auto xOptions = torch::TensorOptions().dtype(torch::kDouble).device(Ms_val.device());
	torch::Tensor retv = torch::zeros({(long)batchSize, (long)M_cols}, xOptions);

	// TODO: do experiments on choice of work group size
	dim3 wgs(1, 16);
	dim3 numBlocks((M_rows + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);

	tmat_vec_kernel<<<numBlocks, wgs>>>(batchSize,
	                                    M_rows,
	                                    M_cols,
	                                    M_nnz,
	                                    M_rowPtr.data_ptr<int>(),
	                                    M_colInd.data_ptr<int>(),
	                                    Ms_val.data_ptr<double>(),
	                                    vec.data_ptr<double>(),
	                                    retv.data_ptr<double>());
	return retv;
}


__global__ void apply_damping_kernel(int batchSize,
                                int M_rows,
                                int M_cols,
                                int M_nnz,
                                const int* M_rowPtr,
                                const int* M_colInd,
                                double* Ms_val,
                                double alpha,
                                double beta) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int batchIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if(batchIndex >= batchSize || row >= M_rows) {
		return;
	}

	int srcRow_offset = M_rowPtr[row];
	int srcRow_len = M_rowPtr[row+1] - srcRow_offset;
	const int* srcRow_colInd = M_colInd + srcRow_offset;
	double* srcRow_val = Ms_val + batchIndex * M_nnz + srcRow_offset;

	for(int i = 0; i < srcRow_len; i++) {
		if(srcRow_colInd[i] == row) {
			srcRow_val[i] += alpha * srcRow_val[i] + beta;
		}
	}
}

void apply_damping(int batchSize,
                   int M_cols,
                   const torch::Tensor& M_rowPtr,
                   const torch::Tensor& M_colInd,
                   const torch::Tensor& Ms_val,
                   double alpha,
                   double beta) {

	int64_t M_rows = M_rowPtr.size(0) - 1;
	int64_t M_nnz = M_colInd.size(0);

	TORCH_CHECK(M_rowPtr.device().is_cuda());
	TORCH_CHECK(M_colInd.device().is_cuda());
	TORCH_CHECK(Ms_val.device().is_cuda());
	TORCH_CHECK(M_rowPtr.dtype() == torch::kInt);
	TORCH_CHECK(M_colInd.dtype() == torch::kInt);
	TORCH_CHECK(Ms_val.dtype() == torch::kDouble); // TODO: add support for float
	TORCH_CHECK(M_rowPtr.dim() == 1);
	TORCH_CHECK(M_colInd.dim() == 1);
	TORCH_CHECK(Ms_val.dim() == 2);
	TORCH_CHECK(Ms_val.size(0) == batchSize);
	TORCH_CHECK(Ms_val.size(1) == M_nnz);

	// TODO: do experiments on choice of work group size
	dim3 wgs(1, 16);
	dim3 numBlocks((M_rows + wgs.x - 1) / wgs.x, (batchSize + wgs.y - 1) / wgs.y);

	apply_damping_kernel<<<numBlocks, wgs>>>(batchSize,
	                                         M_rows,
	                                         M_cols,
	                                         M_nnz,
	                                         M_rowPtr.data_ptr<int>(),
	                                         M_colInd.data_ptr<int>(),
	                                         Ms_val.data_ptr<double>(),
	                                         alpha,
	                                         beta);
}

PYBIND11_MODULE(mat_mult, m) {
    m.doc() = "Python bindings for batched mat operations";
    m.def("mult_MtM", &mult_MtM,
          "Batched multiplication of mat by transpose: Mt * M\n"
          "The sparse structure of the result must be computed\n"
          "beforehand and supplied as MtM_rowPtr, MtM_colInd",
          py::arg("batch_size"),
          py::arg("M_rowPtr"),
          py::arg("M_colInd"),
          py::arg("Ms_val"),
          py::arg("MtM_rowPtr"),
          py::arg("MtM_colInd")
          );
    m.def("mat_vec", &mat_vec,
          "Batched multiplication of mat by vector: M * v",
          py::arg("batch_size"),
          py::arg("M_cols"),
          py::arg("M_rowPtr"),
          py::arg("M_colInd"),
          py::arg("Ms_val"),
          py::arg("vec")
          );
    m.def("tmat_vec", &tmat_vec,
          "Batched multiplication of transposed mat by vector: Mt * v",
          py::arg("batch_size"),
          py::arg("M_cols"),
          py::arg("M_rowPtr"),
          py::arg("M_colInd"),
          py::arg("Ms_val"),
          py::arg("vec")
          );
    m.def("apply_damping", &apply_damping,
          "M.diagonal() += M.diagonal() * alpha + beta",
          py::arg("batch_size"),
          py::arg("M_cols"),
          py::arg("M_rowPtr"),
          py::arg("M_colInd"),
          py::arg("Ms_val"),
          py::arg("alpha"),
          py::arg("beta")
          );
};
