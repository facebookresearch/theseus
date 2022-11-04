# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch
import torch.nn as nn

import theseus.utils as thutils


def test_build_mlp():
    # set seed for mlp
    torch.manual_seed(0)
    # create name to class map for activation function
    act_name_to_cls_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}

    # set test parameters
    test_hidden_depth = [0, 1, 2]
    input_dim = 3
    hidden_dim = 4
    output_dim = 5
    batch_size = 16

    for hidden_depth in test_hidden_depth:
        for act_name in act_name_to_cls_map.keys():
            sample_mlp = thutils.build_mlp(
                input_dim, hidden_dim, output_dim, hidden_depth, act_name
            )
            # check depth by counting linear modules
            layer_count = 0
            for curr_mod in sample_mlp:
                if isinstance(curr_mod, nn.Linear):
                    layer_count += 1
            created_depth = layer_count - 1
            assert (
                created_depth == hidden_depth
            ), f"Incorrect number of layers: {created_depth} should be {hidden_depth}"

            # check input and output sizes
            x = torch.rand(batch_size, input_dim)
            y = sample_mlp(x)  # automatically tests input size
            assert (
                y.size(0) == batch_size and y.size(1) == output_dim
            ), "Incorrect output tensor created"

            # check size of each layer in mlp
            def _get_layer_sizes(layer_id):
                layer_input_dim, layer_output_dim = hidden_dim, hidden_dim
                if layer_id == 0:
                    layer_input_dim = input_dim
                if layer_id == created_depth:
                    layer_output_dim = output_dim
                return layer_input_dim, layer_output_dim

            layer_id = 0
            for curr_mod in sample_mlp:
                if isinstance(curr_mod, nn.Linear):
                    layer_input_dim, layer_output_dim = _get_layer_sizes(layer_id)
                    x = torch.rand(batch_size, layer_input_dim)
                    y = curr_mod(x)
                    assert (
                        y.size(0) == batch_size and y.size(1) == layer_output_dim
                    ), f"Incorrect hidden layer dimensions at layer {layer_id}"
                    layer_id += 1

            # check activation function
            # assume all non Linear layers must be activation layers
            act_cls = act_name_to_cls_map[act_name]
            for curr_mod in sample_mlp:
                if not isinstance(curr_mod, nn.Linear):
                    assert isinstance(
                        curr_mod, act_cls
                    ), f"Incorrect activation class: {curr_mod} should be {act_cls}"


def test_gather_from_rows_cols():
    rng = np.random.default_rng(0)
    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(100):
        batch_size = rng.integers(1, 8)
        num_rows = rng.integers(1, 20)
        num_cols = rng.integers(1, 20)
        num_points = rng.integers(1, 100)
        matrix = torch.randn((batch_size, num_rows, num_cols))
        rows = torch.randint(0, num_rows, size=(batch_size, num_points))
        cols = torch.randint(0, num_cols, size=(batch_size, num_points))
        res = thutils.gather_from_rows_cols(matrix, rows, cols)
        assert res.shape == (batch_size, num_points)
        for i in range(batch_size):
            for j in range(num_points):
                assert torch.allclose(res[i, j], matrix[i, rows[i, j], cols[i, j]])
