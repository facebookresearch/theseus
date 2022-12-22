# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import theseus as th


def random_scalar(batch_size):
    number = torch.rand(1).item()
    # Test with all possible cell_size inputs
    if number < 1.0 / 3:
        return th.Variable(tensor=torch.randn(batch_size, 1))
    elif number < 2.0 / 3:
        return torch.randn(batch_size, 1)
    else:
        return torch.randn(1).item()


def random_origin(batch_size):
    origin_tensor = torch.randn(batch_size, 2)
    if torch.rand(1).item() < 0.5:
        return th.Point2(tensor=origin_tensor)
    return origin_tensor


def random_sdf_data(batch_size, field_width, field_height):
    sdf_data_tensor = torch.randn(batch_size, field_width, field_height)
    if torch.rand(1).item() < 0.5:
        return th.Variable(tensor=sdf_data_tensor)
    return sdf_data_tensor


def random_sdf(batch_size, field_width, field_height):
    return th.eb.SignedDistanceField2D(
        random_origin(batch_size),
        random_scalar(batch_size),
        random_sdf_data(batch_size, field_width, field_height),
    )
