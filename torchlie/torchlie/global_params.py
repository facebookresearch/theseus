# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Any, Dict

import torch


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            f"Unsupported data type {dtype}. "
            "Theseus only supports 32- and 64-bit tensors."
        )


@dataclass
class _TorchLieGlobalParams:
    so3_near_pi_eps_float32: float = 0
    so3_near_zero_eps_float32: float = 0
    so3_matrix_eps_float32: float = 0
    so3_quat_eps_float32: float = 0
    so3_hat_eps_float32: float = 0
    se3_hat_eps_float32: float = 0
    so3_near_pi_eps_float64: float = 0
    so3_near_zero_eps_float64: float = 0
    so3_matrix_eps_float64: float = 0
    so3_quat_eps_float64: float = 0
    so3_hat_eps_float64: float = 0
    se3_hat_eps_float64: float = 0
    _allow_passthrough_ops: bool = False
    _faster_log_maps: bool = False

    def __init__(self):
        self.reset()

    def get_eps(self, ltype: str, attr: str, dtype: torch.dtype) -> float:
        _CHECK_DTYPE_SUPPORTED(dtype)
        attr_name = f"{ltype}_{attr}_eps_{str(dtype)[6:]}"
        return getattr(self, attr_name)

    def reset(self) -> None:
        self.so3_near_pi_eps_float32 = 1e-2
        self.so3_near_zero_eps_float32 = 1e-2
        self.so3_d_near_zero_eps_float32 = 2e-1
        self.so3_matrix_eps_float32 = 4e-4
        self.so3_quat_eps_float32 = 2e-4
        self.so3_hat_eps_float32 = 5e-6
        self.se3_hat_eps_float32 = 5e-6
        self.so3_near_pi_eps_float64 = 1e-7
        self.so3_near_zero_eps_float64 = 5e-3
        self.so3_d_near_zero_eps_float64 = 1e-2
        self.so3_matrix_eps_float64 = 1e-6
        self.so3_quat_eps_float64 = 5e-7
        self.so3_hat_eps_float64 = 5e-7
        self.se3_hat_eps_float64 = 5e-7


_TORCHLIE_GLOBAL_PARAMS = _TorchLieGlobalParams()


def set_global_params(options: Dict[str, Any]) -> None:
    for k, v in options.items():
        if not hasattr(_TORCHLIE_GLOBAL_PARAMS, k):
            raise ValueError(f"{k} is not a valid global option for torchlie.")
        setattr(_TORCHLIE_GLOBAL_PARAMS, k, v)


def reset_global_params() -> None:
    _TORCHLIE_GLOBAL_PARAMS.reset()
