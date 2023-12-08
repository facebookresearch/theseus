# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
import torchlie as lie
import torchlie.functional.se3_impl as se3_impl
import torchlie.functional.so3_impl as so3_impl

from .functional.common import get_test_cfg, sample_inputs


@pytest.fixture
def rng():
    rng_ = torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
    rng_.manual_seed(0)
    return rng_


# Converts the functional sampled inputs to the class-based inputs
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
        return (x,)
    y = _get_typed_tensor(1)
    return (x, y)


@pytest.mark.parametrize(
    "op_name",
    [
        "exp",
        "hat",
        "vee",
        "lift",
        "project",
        "compose",
        "transform",
        "untransform",
        "left_act",
        "left_project",
        "log",
        "inv",
        "adj",
    ],
)
@pytest.mark.parametrize("ltype_str", ["se3", "so3"])
@pytest.mark.parametrize("batch_size", [5])
def test_op(op_name, ltype_str, batch_size, rng):
    ltype = {"se3": lie.SE3, "so3": lie.SO3}[ltype_str]

    def _get_impl(ltype):
        return {lie.SE3: se3_impl, lie.SO3: so3_impl}[ltype]

    def _to_functional_fmt(x):
        def _to_torch(t):
            return t._t if isinstance(t, lie.LieTensor) else t

        if isinstance(x, tuple):  # jacobians output
            return x[0], _to_torch(x[1])
        return _to_torch(x)

    aux_name = op_name if op_name in ["inv", "adj"] else "other"
    # This is needed because the backend implementation has a different name
    # (these are not publicly exposed).
    impl_name = {"other": op_name, "inv": "inverse", "adj": "adjoint"}[aux_name]
    dim = {lie.SE3: 6, lie.SO3: 3}[ltype]
    data_shape = {lie.SE3: (3, 4), lie.SO3: (3, 3)}[ltype]
    impl_module = _get_impl(ltype)
    all_input_types, _ = get_test_cfg(
        impl_name, torch.float32, dim, data_shape, module=impl_module
    )
    for input_types in all_input_types:
        inputs = sample_inputs(input_types, batch_size, torch.float32, rng)
        lie_tensor_inputs = _get_lie_tensor_inputs(input_types, inputs, ltype)
        fn_holder_object = ltype if hasattr(ltype, op_name) else lie
        out = _to_functional_fmt(getattr(fn_holder_object, op_name)(*lie_tensor_inputs))
        impl_out = getattr(impl_module, f"_{impl_name}_autograd_fn")(*inputs)
        torch.testing.assert_close(out, impl_out)

        # Also check that class-version is correct
        if hasattr(ltype, op_name):
            continue  # No class version for things like exp, hat, vee, etc.

        # Use a dummy group for static ops (e.g., exp, hat)
        c = (
            ltype.rand(1, generator=rng, dtype=torch.float32)
            if isinstance(lie_tensor_inputs[-1], lie.ltype)
            else lie_tensor_inputs[0]
        )
        c_inputs = (
            ()
            if len(lie_tensor_inputs) == 1
            else (
                lie_tensor_inputs[0]
                if isinstance(lie_tensor_inputs[1], lie.ltype)
                else lie_tensor_inputs[1],
            )
        )
        out_c = _to_functional_fmt(getattr(c, op_name)(*c_inputs))
        torch.testing.assert_close(out, out_c)

    if op_name in ["exp", "compose", "log", "inv", "transform", "untransform"]:
        fn_holder_object = ltype if op_name == "exp" else lie
        jac1, out = _to_functional_fmt(
            getattr(fn_holder_object, f"j{op_name}")(*lie_tensor_inputs)
        )
        impl_jac, impl_out = getattr(impl_module, f"_j{impl_name}_autograd_fn")(*inputs)
        torch.testing.assert_close([jac1, out], [impl_jac, impl_out])

        # Check class-version (exp doesn't have a class version)
        if op_name != "exp":
            jac_c, out_c = _to_functional_fmt(getattr(c, f"j{op_name}")(*c_inputs))
            torch.testing.assert_close([jac1, out], [jac_c, out_c])


def test_backward_works():
    # Runs optimization to check that the compute graph is not broken
    def _check(opt_tensor, target_tensor, tensor_fn):
        opt = torch.optim.Adam([opt_tensor])
        losses = []
        for i in range(2):
            opt.zero_grad()
            d = tensor_fn(opt_tensor).local(target_tensor)
            loss = torch.sum(d**2)
            losses.append(loss.detach().clone())
            loss.backward()
            opt.step()
        assert not losses[0].allclose(losses[-1])

    # Check local op from a random tensor
    g1 = lie.SE3.rand(1, requires_grad=True)
    g2 = lie.SE3.rand(1)
    _check(g1, g2, lambda x: x)

    # Check local op from exp map
    vec = torch.randn((2, 6), requires_grad=True)
    eye = lie.SE3.identity(2)
    _check(vec, eye, lambda x: lie.SE3.exp(x))
