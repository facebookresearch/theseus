# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

import theseus.labs.lie as lie
import theseus.labs.lie.functional.se3_impl as se3_impl
import theseus.labs.lie.functional.so3_impl as so3_impl

from .functional.common import get_test_cfg, sample_inputs


def _get_impl(ltype):
    return {lie.SE3: se3_impl, lie.SO3: so3_impl}[ltype]


@pytest.fixture
def rng():
    rng_ = torch.Generator()
    rng_.manual_seed(0)
    return rng_


@pytest.mark.parametrize("op_name", ["log", "inv", "adj"])
@pytest.mark.parametrize("ltype", [lie.SE3, lie.SO3])
@pytest.mark.parametrize("batch_size", [5])
def test_op_no_args(op_name, ltype, batch_size, rng):
    impl_name = {"log": "log", "inv": "inverse", "adj": "adjoint"}[op_name]
    out_is_group = op_name == "inv"
    impl_module = _get_impl(ltype)
    x = lie.rand(ltype, batch_size, generator=rng)
    out = getattr(x, f"{op_name}")()
    out = out._t if out_is_group else out
    impl_out = getattr(impl_module, f"_{impl_name}_autograd_fn")(x._t)
    torch.testing.assert_close(out, impl_out)

    if op_name in ["log", "inv"]:
        out1, out2 = getattr(x, f"j{op_name}")()
        out2 = out2._t if out_is_group else out2
        impl_out1, impl_out2 = getattr(impl_module, f"_j{impl_name}_autograd_fn")(x._t)
        torch.testing.assert_close(out1, impl_out1)
        torch.testing.assert_close(out2, impl_out2)


def _get_lie_tensor_inputs(input_types, sampled_inputs, ltype):
    def _get_typed_tensor(idx):
        is_group = input_types[idx][0] == "group"
        return (
            lie.LieTensor(sampled_inputs[idx], ltype)
            if is_group
            else sampled_inputs[idx]
        )

    x = _get_typed_tensor(0)
    if len(sampled_inputs) == 1:
        # For static method (exp, hat, vee, lift, project), we need to
        # specify the ltype as the first input
        return (x,) if input_types[0][0] == "group" else (ltype, x)
    y = _get_typed_tensor(1)
    return (x, y)


@pytest.mark.parametrize(
    "op_name",
    ["exp", "hat", "vee", "lift", "project", "compose", "left_act", "left_project"],
)
@pytest.mark.parametrize("ltype", [lie.SE3, lie.SO3])
@pytest.mark.parametrize("batch_size", [5])
def test_op(op_name, ltype, batch_size, rng):
    dim = {lie.SE3: 6, lie.SO3: 3}[ltype]
    out_is_group = op_name in ["exp", "compose"]
    data_shape = {lie.SE3: (3, 4), lie.SO3: (3, 3)}[ltype]
    impl_module = _get_impl(ltype)
    all_input_types, _ = get_test_cfg(
        op_name, torch.float32, dim, data_shape, module=impl_module
    )
    for input_types in all_input_types:
        inputs = sample_inputs(input_types, batch_size, torch.float32, rng)
        lie_tensor_inputs = _get_lie_tensor_inputs(input_types, inputs, ltype)
        out = getattr(lie, op_name)(*lie_tensor_inputs)
        out = out._t if out_is_group else out
        impl_out = getattr(impl_module, f"_{op_name}_autograd_fn")(*inputs)
        torch.testing.assert_close(out, impl_out)

    if op_name in ["exp", "compose"]:
        out1, out2 = getattr(lie, f"j{op_name}")(*lie_tensor_inputs)
        out2 = out2._t if out_is_group else out2
        impl_out1, impl_out2 = getattr(impl_module, f"_j{op_name}_autograd_fn")(*inputs)
        torch.testing.assert_close(out1, impl_out1)
        torch.testing.assert_close(out2, impl_out2)
