# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict
from dataclasses import dataclass

import lie.options
import torch


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            f"Unsupported data type {dtype}. "
            "Theseus only supports 32- and 64-bit tensors."
        )


@dataclass
class _TheseusGlobalOptions:
    so2_norm_eps_float32: float = 0
    so2_matrix_eps_float32: float = 0
    se2_near_zero_eps_float32: float = 0
    so3_near_pi_eps_float32: float = 0
    so3_near_zero_eps_float32: float = 0
    so2_norm_eps_float64: float = 0
    so2_matrix_eps_float64: float = 0
    se2_near_zero_eps_float64: float = 0
    so3_near_pi_eps_float64: float = 0
    so3_near_zero_eps_float64: float = 0

    def __init__(self):
        self.reset()

    def get_eps(self, ltype: str, attr: str, dtype: torch.dtype) -> float:
        _CHECK_DTYPE_SUPPORTED(dtype)
        attr_name = f"{ltype}_{attr}_eps_{str(dtype)[6:]}"
        return getattr(self, attr_name)

    def reset(self) -> None:
        self.so2_norm_eps_float32 = 1e-12
        self.so2_matrix_eps_float32 = 1e-5
        self.se2_near_zero_eps_float32 = 3e-2
        self.so3_near_pi_eps_float32 = 1e-2
        self.so3_near_zero_eps_float32 = 1e-2

        self.so2_norm_eps_float32 = 1e-12
        self.so2_matrix_eps_float64 = 4e-7
        self.se2_near_zero_eps_float64 = 1e-6
        self.so3_near_pi_eps_float64 = 1e-7
        self.so3_near_zero_eps_float64 = 5e-3


_THESEUS_GLOBAL_OPTIONS = _TheseusGlobalOptions()


_TORCHLIE_PREFIX = "torchlie."


def set_global_options(options: Dict[str, Any]) -> None:
    torchlie_options = {
        k.lstrip(_TORCHLIE_PREFIX): v
        for k, v in options.items()
        if _TORCHLIE_PREFIX in k
    }
    theseus_options = {
        k: v for k, v in options.items() if _TORCHLIE_PREFIX not in options
    }
    for k, v in theseus_options.items():
        if not hasattr(_THESEUS_GLOBAL_OPTIONS, k):
            raise ValueError(f"{k} is not a valid global option for theseus.")
        setattr(_THESEUS_GLOBAL_OPTIONS, k, v)
    lie.options.set_global_options(torchlie_options)
