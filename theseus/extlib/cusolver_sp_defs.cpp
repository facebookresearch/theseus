// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "cusolver_sp_defs.h"
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <ATen/cuda/CUDAContext.h>

// functions are defined in this headers are inline so this can be included multiple times
// in units compiled independently (such as Torch extensions formed by one .cu/.cpp file)
namespace theseus::cusolver_sp {

	const char* cusolverGetErrorMessage(cusolverStatus_t status) {
		switch (status) {
		case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCES";
		case CUSOLVER_STATUS_NOT_INITIALIZED:             return "CUSOLVER_STATUS_NOT_INITIALIZED";
		case CUSOLVER_STATUS_ALLOC_FAILED:                return "CUSOLVER_STATUS_ALLOC_FAILED";
		case CUSOLVER_STATUS_INVALID_VALUE:               return "CUSOLVER_STATUS_INVALID_VALUE";
		case CUSOLVER_STATUS_ARCH_MISMATCH:               return "CUSOLVER_STATUS_ARCH_MISMATCH";
		case CUSOLVER_STATUS_EXECUTION_FAILED:            return "CUSOLVER_STATUS_EXECUTION_FAILED";
		case CUSOLVER_STATUS_INTERNAL_ERROR:              return "CUSOLVER_STATUS_INTERNAL_ERROR";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
		default:                                          return "Unknown cusolver error number";
		}
	}

	void createCusolverSpHandle(cusolverSpHandle_t *handle) {
		CUSOLVER_CHECK(cusolverSpCreate(handle));
	}

	// The switch below look weird, but we will be adopting the same policy as for CusolverDn handle in Torch source
	void destroyCusolverSpHandle(cusolverSpHandle_t handle) {
		// this is because of something dumb in the ordering of
		// destruction. Sometimes atexit, the cuda context (or something)
		// would already be destroyed by the time this gets destroyed. It
		// happens in fbcode setting. @colesbury and @soumith decided to not destroy
		// the handle as a workaround.
		//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
		cusolverSpDestroy(handle);
#endif
	}

	using CuSolverSpPoolType = at::cuda::DeviceThreadHandlePool<cusolverSpHandle_t, createCusolverSpHandle, destroyCusolverSpHandle>;

	cusolverSpHandle_t getCurrentCUDASolverSpHandle() {
		int device;
		AT_CUDA_CHECK(cudaGetDevice(&device));

		// Thread local PoolWindows are lazily-initialized
		// to avoid initialization issues that caused hangs on Windows.
		// See: https://github.com/pytorch/pytorch/pull/22405
		// This thread local unique_ptrs will be destroyed when the thread terminates,
		// releasing its reserved handles back to the pool.
		static auto pool = std::make_shared<CuSolverSpPoolType>();
		thread_local std::unique_ptr<CuSolverSpPoolType::PoolWindow> myPoolWindow(pool->newPoolWindow());

		auto handle = myPoolWindow->reserve(device);
		auto stream = c10::cuda::getCurrentCUDAStream();
		CUSOLVER_CHECK(cusolverSpSetStream(handle, stream));
		return handle;
	}

} // namespace theseus::cusolver_sp
