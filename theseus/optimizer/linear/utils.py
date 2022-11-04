# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union
import torch


# See Nocedal and Wright, Numerical Optimization, pp. 260 and 261
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
def convert_to_alpha_beta_damping_tensors(
    damping: Union[float, torch.Tensor],
    damping_eps: float,
    ellipsoidal_damping: bool,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    damping = torch.as_tensor(damping).to(device=device, dtype=dtype)
    if damping.ndim > 1:
        raise ValueError("Damping must be a float or a 1-D tensor.")
    if damping.ndim == 0 or damping.shape[0] == 1 and batch_size != 1:
        # Our damp kernel does not like expand, since it may try
        # to access indices beyond what's actually stored in this tensor
        damping = damping.repeat(batch_size)
    return (
        (damping, damping_eps * torch.ones_like(damping))
        if ellipsoidal_damping
        else (torch.zeros_like(damping), damping)
    )
