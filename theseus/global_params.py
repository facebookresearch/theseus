# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, fields
from typing import Any, Dict

import torch

from torchlie.global_params import _TORCHLIE_GLOBAL_PARAMS as _LIE_GP

_LIE_GP_FIELD_NAMES = set([f.name for f in fields(_LIE_GP)])


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            f"Unsupported data type {dtype}. "
            "Theseus only supports 32- and 64-bit tensors."
        )


@dataclass
class _TheseusGlobalParams:
    so2_norm_eps_float32: float = 0
    so2_matrix_eps_float32: float = 0
    se2_near_zero_eps_float32: float = 0
    so3_to_quaternion_sqrt_eps_float32: float = 0
    so2_norm_eps_float64: float = 0
    so2_matrix_eps_float64: float = 0
    se2_near_zero_eps_float64: float = 0
    so3_to_quaternion_sqrt_eps_float64: float = 0
    fast_approx_local_jacobians: bool = False

    def __init__(self):
        self.reset()

    def get_eps(self, ltype: str, attr: str, dtype: torch.dtype) -> float:
        try:
            return _LIE_GP.get_eps(ltype, attr, dtype)
        except AttributeError:
            _CHECK_DTYPE_SUPPORTED(dtype)
            attr_name = f"{ltype}_{attr}_eps_{str(dtype)[6:]}"
            return getattr(self, attr_name)

    def reset(self) -> None:
        self.so2_norm_eps_float32 = 1e-12
        self.so2_matrix_eps_float32 = 1e-5
        self.se2_near_zero_eps_float32 = 3e-2
        self.se2_d_near_zero_eps_float32 = 1e-1
        self.so3_to_quaternion_sqrt_eps_float32 = 1e-6
        self.so2_norm_eps_float32 = 1e-12

        self.so2_matrix_eps_float64 = 4e-7
        self.se2_near_zero_eps_float64 = 1e-6
        self.se2_d_near_zero_eps_float64 = 1e-3
        self.so3_to_quaternion_sqrt_eps_float64 = 1e-6

        self.fast_approx_local_jacobians = False


_THESEUS_GLOBAL_PARAMS = _TheseusGlobalParams()


def set_global_params(options: Dict[str, Any]) -> None:
    torchlie_params_found = []
    for k in options:
        if k in _LIE_GP_FIELD_NAMES:
            torchlie_params_found.append(k)
    if torchlie_params_found:
        raise RuntimeError(
            f"Theseus uses torchlie for configuring 3D Lie group tolerances, "
            f"but you attempted to use theseus.set_global_params() for the "
            f"following ones:\n    {torchlie_params_found}.\n"
            f"Please use torchlie.set_global_params() to set these tolerances."
        )
    for k, v in options.items():
        if not hasattr(_THESEUS_GLOBAL_PARAMS, k):
            raise ValueError(f"{k} is not a valid global option for theseus.")
        setattr(_THESEUS_GLOBAL_PARAMS, k, v)
