# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch


def check_jacobians_list(jacobians: List[torch.Tensor]):
    if len(jacobians) != 0:
        raise ValueError("jacobians list to be populated must be empty.")


def get_module(module_name):
    return __import__(module_name, fromlist=[""])


def get_cls_module(cls):
    return get_module(cls.__module__)


def shape_err_msg(data_type: str, expected_shape: str, tensor_shape: torch.Size) -> str:
    return f"{data_type} must have shape {expected_shape} but got shape {tensor_shape}."


def fill_dims(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor.view((1,) * (dim - tensor.ndim) + tensor.shape)


def permute_op_dim(dim: int, op_dim: int, group_dim: int):
    return (
        [*range(dim - op_dim - group_dim, dim - group_dim)]
        + [*range(0, dim - op_dim - group_dim)]
        + [*range(dim - group_dim, dim)]
    )


def unpermute_op_dim(dim: int, op_dim: int, group_dim: int):
    return (
        [*range(op_dim, dim - group_dim)]
        + [*range(0, op_dim)]
        + [*range(dim - group_dim, dim)]
    )
