# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import numpy as np
import torch


def compute_A_grad(
    batch_size: int,
    A_row_ptr: np.ndarray,
    A_col_ind: np.ndarray,
    b: torch.Tensor,
    x: torch.Tensor,
    b_Ax: torch.Tensor,
    H: torch.Tensor,
    AH: torch.Tensor,
    damping_alpha_beta: Optional[Tuple[torch.Tensor, torch.Tensor]],
    A_val: Optional[torch.Tensor],  # only needed if damping passed
    ctx_A_col_ind: Optional[torch.Tensor],  # only needed if damping passed
    detach_hessian: bool,
):
    A_grad = torch.empty(
        size=(batch_size, len(A_col_ind)), device=x.device
    )  # return value, A's grad
    for r in range(len(A_row_ptr) - 1):
        start, end = A_row_ptr[r], A_row_ptr[r + 1]
        columns = A_col_ind[start:end]  # col indices, for this row
        if detach_hessian:
            A_grad[:, start:end] = b[:, r].unsqueeze(1) * H[:, columns]
        else:
            A_grad[:, start:end] = (
                b_Ax[:, r].unsqueeze(1) * H[:, columns]
                - AH[:, r].unsqueeze(1) * x[:, columns]
            )

    # apply correction if there is a multiplicative damping
    if damping_alpha_beta is not None and (damping_alpha_beta[0] > 0.0).any():
        assert (
            not detach_hessian
        )  # this should only be used with a GN-step with no damping
        alpha = damping_alpha_beta[0].view(-1, 1)
        alpha2Hx = (alpha * 2.0) * H * x  # componentwise product
        A_grad -= A_val * alpha2Hx[:, ctx_A_col_ind.type(torch.long)]

    return A_grad
