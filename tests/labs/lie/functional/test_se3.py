# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.decorators import run_if_labs
from .common import BATCH_SIZES_TO_TEST, TEST_EPS, check_lie_group_function, run_test_op


@run_if_labs()
@pytest.mark.parametrize(
    "op_name",
    [
        "exp",
        "log",
        "adjoint",
        "inverse",
        "hat",
        "compose",
        "transform_from",
        "lift",
        "project",
        "left_act",
        "left_project",
    ],
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_op(op_name, batch_size, dtype):
    import theseus.labs.lie.functional.se3_impl as SE3

    rng = torch.Generator()
    rng.manual_seed(0)
    run_test_op(op_name, batch_size, dtype, rng, 6, (3, 4), SE3)


@run_if_labs()
@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie.functional.se3_impl as SE3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 6, dtype=dtype, generator=rng)
    matrix = SE3._hat_autograd_fn(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(SE3, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = SE3._vee_autograd_fn(matrix)
    torch.testing.assert_close(
        actual_tangent_vector, tangent_vector, atol=TEST_EPS, rtol=TEST_EPS
    )


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["exp", "inv"])
def test_jacrev_unary(batch_size, name):
    if not hasattr(torch, "vmap"):
        return

    import theseus.labs.lie.functional as lieF

    test_fn = getattr(lieF.SE3, name)

    fn_input = (
        torch.randn(batch_size, 6) if name == "exp" else lieF.SE3.rand(batch_size)
    )

    def f(t):
        return lieF.SE3.log(test_fn(t.unsqueeze(0))).squeeze(0)

    j = torch.vmap(torch.func.jacrev(f))(fn_input)
    jac_vmap = lieF.SE3.left_project(fn_input, j) if name != "exp" else j

    jlog, jtest = [], []
    lieF.SE3.log(test_fn(fn_input, jacobians=jtest), jacobians=jlog)
    jac_analytic = jlog[0] @ jtest[0]

    torch.testing.assert_close(jac_vmap, jac_analytic)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["compose"])
def test_jacrev_binary(batch_size, name):
    if not hasattr(torch, "vmap"):
        return

    import theseus.labs.lie.functional as lieF

    test_fn = getattr(lieF.SE3, name)

    fn_inputs = (
        (lieF.SE3.rand(batch_size), lieF.SE3.rand(batch_size))
        if name == "compose"
        else (lieF.SE3.rand(batch_size), torch.randn(batch_size, 3))
    )

    def f(t1, t2):
        op_out = test_fn(t1.unsqueeze(0), t2.unsqueeze(0))
        if name == "compose":
            op_out = lieF.SE3.log(op_out)
        return op_out.squeeze(0)

    jacs_vmap = []
    for i in range(2):
        j = torch.vmap(torch.func.jacrev(f, i))(fn_inputs[0], fn_inputs[1])
        if fn_inputs[i].ndim == 3:  # group input
            j = lieF.SE3.left_project(fn_inputs[i], j)
        jacs_vmap.append(j)

    jlog, jtest = [], []
    out = test_fn(fn_inputs[0], fn_inputs[1], jacobians=jtest)
    if name == "compose":
        lieF.SE3.log(out, jacobians=jlog)
    for i in range(2):
        jac_analytic = jlog[0] @ jtest[i] if name == "compose" else jtest[i]
        torch.testing.assert_close(jacs_vmap[i], jac_analytic)
