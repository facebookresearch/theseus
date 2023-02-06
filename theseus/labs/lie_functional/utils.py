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
