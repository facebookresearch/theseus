// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cusolverSp.h>

#define CUSOLVER_CHECK(EXPR)                                            \
	do { \
		cusolverStatus_t __err = EXPR; \
		TORCH_CHECK(__err == CUSOLVER_STATUS_SUCCESS, \
		            "cusolver error: ", \
		            theseus::cusolver_sp::cusolverGetErrorMessage(__err), \
		            ", when calling `" #EXPR "`"); \
	} while (0)

namespace theseus::cusolver_sp {

	const char* cusolverGetErrorMessage(cusolverStatus_t status);

	cusolverSpHandle_t getCurrentCUDASolverSpHandle();

} // namespace theseus::cusolver_sp
