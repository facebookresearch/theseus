# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import scipy.sparse
import torch
import torch.nn as nn

import theseus as th
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


def _check_sparse_mv_and_mtv(batch_size, num_rows, num_cols, fill, device):
    A_col_ind, A_row_ptr, A_val, _ = thutils.random_sparse_matrix(
        batch_size,
        num_rows,
        num_cols,
        fill,
        min(num_cols, 2),
        torch.Generator(device),
        device,
    )
    b = torch.randn(batch_size, num_cols, device=device).double()
    b2 = torch.randn(batch_size, num_rows, device=device).double()

    # Check backward pass
    if batch_size < 16:
        A_val.requires_grad = True
        b.requires_grad = True
        torch.autograd.gradcheck(
            thutils.sparse_mv, (num_cols, A_row_ptr, A_col_ind, A_val, b)
        )
        torch.autograd.gradcheck(
            thutils.sparse_mtv, (num_cols, A_row_ptr, A_col_ind, A_val, b2)
        )

    # Check forward pass
    Ab_bundle = [scipy.sparse.csr_matrix, thutils.sparse_mv, (num_rows, num_cols), b]
    Atb_bundle = [scipy.sparse.csc_matrix, thutils.sparse_mtv, (num_cols, num_rows), b2]
    for i in range(batch_size):
        for sparse_constructor, sparse_op, shape, b_tensor in [Ab_bundle, Atb_bundle]:
            out = sparse_op(num_cols, A_row_ptr, A_col_ind, A_val, b_tensor)
            A_sparse = sparse_constructor(
                (
                    A_val[i].detach().cpu().numpy(),
                    A_col_ind.cpu().numpy(),
                    A_row_ptr.cpu().numpy(),
                ),
                shape,
            )
            expected_out = A_sparse * b_tensor[i].detach().cpu().numpy()
            diff = expected_out - out[i].detach().cpu().numpy()
            assert np.linalg.norm(diff) < 1e-8


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("num_rows", [1, 32])
@pytest.mark.parametrize("num_cols", [1, 4, 32])
@pytest.mark.parametrize("fill", [0.1, 0.9])
def test_sparse_mv_cpu(batch_size, num_rows, num_cols, fill):
    _check_sparse_mv_and_mtv(batch_size, num_rows, num_cols, fill, "cpu")


@pytest.mark.cudaext
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("num_rows", [1, 32])
@pytest.mark.parametrize("num_cols", [1, 4, 32])
@pytest.mark.parametrize("fill", [0.1, 0.9])
def test_sparse_mv_cuda(batch_size, num_rows, num_cols, fill):
    _check_sparse_mv_and_mtv(batch_size, num_rows, num_cols, fill, "cuda:0")


def test_jacobians_check():
    se3s = [th.SE3() for _ in range(3)]
    w = th.ScaleCostWeight(0.5)
    cf = th.Difference(se3s[0], se3s[1], w)
    thutils.check_jacobians(cf, 1)

    cf = th.Between(se3s[0], se3s[1], se3s[2], w)
    thutils.check_jacobians(cf, 1)

    cf = th.eb.DoubleIntegrator(se3s[0], th.Vector(6), se3s[1], th.Vector(6), 1.0, w)
    thutils.check_jacobians(cf, 1)


def test_timer():
    # Check different ways of instantiating work correctly
    with thutils.Timer("cpu") as timer:
        torch.randn(1)
    assert timer.elapsed_time > 0

    with thutils.Timer("cpu", active=False) as timer:
        torch.randn(1)
    assert timer.elapsed_time == 0

    timer = thutils.Timer("cpu")
    with timer:
        torch.randn(1)
    assert timer.elapsed_time > 0

    timer = thutils.Timer("cpu", active=False)
    with timer:
        torch.randn(1)
    assert timer.elapsed_time == 0

    # Checking that deactivate keyword works correctly
    timer = thutils.Timer("cpu")
    with timer("randn", deactivate=True):
        torch.randn(1)
    assert timer.elapsed_time == 0
    timer.start("randn", deactivate=True)
    torch.randn(1)
    timer.end()
    assert timer.elapsed_time == 0
    # Checking that stats accumulation works correctly
    with timer("randn"):
        torch.randn(1)
    timer.start("randn")
    torch.randn(1)
    timer.end()
    with timer("mult"):
        torch.ones(1) * torch.zeros(1)
    stats = timer.stats()
    assert "randn" in stats and "mult" in stats
    assert len(stats["randn"]) == 2
    assert len(stats["mult"]) == 1
