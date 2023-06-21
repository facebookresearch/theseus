# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict
from dataclasses import dataclass

import torch


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            f"Unsupported data type {dtype}. "
            "Theseus only supports 32- and 64-bit tensors."
        )


@dataclass
class _TorchLieOptions:
    so2_norm_eps_float32: float = 1e-12
    so2_matrix_eps_float32: float = 1e-5
    se2_near_zero_eps_float32: float = 3e-2
    so3_near_pi_eps_float32: float = 1e-2
    so3_near_zero_eps_float32: float = 1e-2
    so3_matrix_eps_float32: float = 4e-4
    so3_quat_eps_float32: float = 2e-4
    so3_hat_eps_float32: float = 5e-6
    se3_near_pi_eps_float32: float = 1e-2
    se3_near_zero_eps_float32: float = 1e-2
    se3_hat_eps_float32: float = 5e-6
    so2_norm_eps_float64: float = 1e-12
    so2_matrix_eps_float64: float = 4e-7
    se2_near_zero_eps_float64: float = 1e-6
    so3_near_pi_eps_float64: float = 1e-7
    so3_near_zero_eps_float64: float = 5e-3
    so3_matrix_eps_float64: float = 1e-6
    so3_quat_eps_float64: float = 5e-7
    so3_hat_eps_float64: float = 5e-7
    se3_near_pi_eps_float64: float = 1e-7
    se3_near_zero_eps_float64: float = 5e-3
    se3_hat_eps_float64: float = 5e-7

    def get_eps(self, ltype: str, attr: str, dtype: torch.dtype) -> float:
        _CHECK_DTYPE_SUPPORTED(dtype)
        attr_name = f"{ltype}_{attr}_eps_{str(dtype)[6:]}"
        return getattr(self, attr_name)


_TORCHLIE_GLOBAL_OPTIONS = _TorchLieOptions()


def set_global_options(options: Dict[str, Any]) -> None:
    for k, v in options.items():
        if not hasattr(_TORCHLIE_GLOBAL_OPTIONS, k):
            raise ValueError(f"{k} is not a valid global option for torchlie.")
        setattr(_TORCHLIE_GLOBAL_OPTIONS, k, v)
