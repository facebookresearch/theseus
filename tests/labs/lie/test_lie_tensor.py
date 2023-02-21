# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.decorators import run_if_labs
from .functional.common import get_test_cfg, sample_inputs


@pytest.fixture
def rng():
    rng_ = torch.Generator()
    rng_.manual_seed(0)
    return rng_


# Converts the functional sampled inputs to the class-based inputs
def _get_lie_tensor_inputs(input_types, sampled_inputs, ltype):
    import theseus.labs.lie as lie

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


@run_if_labs()
@pytest.mark.parametrize(
    "op_name",
    [
        "exp",
        "hat",
        "vee",
        "lift",
        "project",
        "compose",
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
    import theseus.labs.lie as lie
    import theseus.labs.lie.functional.se3_impl as se3_impl
    import theseus.labs.lie.functional.so3_impl as so3_impl

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
        out = _to_functional_fmt(getattr(lie, op_name)(*lie_tensor_inputs))
        impl_out = getattr(impl_module, f"_{impl_name}_autograd_fn")(*inputs)
        torch.testing.assert_close(out, impl_out)

        # Also check that class-version is correct
        # Use a dummy group for static ops (e.g., exp, hat)
        c = (
            lie.rand(1, ltype, generator=rng, dtype=torch.float32)
            if isinstance(lie_tensor_inputs[0], lie.ltype)
            else lie_tensor_inputs[0]
        )
        c_inputs = () if len(lie_tensor_inputs) == 1 else (lie_tensor_inputs[1],)
        out_c = _to_functional_fmt(getattr(c, op_name)(*c_inputs))
        torch.testing.assert_close(out, out_c)

    if op_name in ["exp", "compose", "log", "inv"]:
        jac1, out = _to_functional_fmt(getattr(lie, f"j{op_name}")(*lie_tensor_inputs))
        impl_jac, impl_out = getattr(impl_module, f"_j{impl_name}_autograd_fn")(*inputs)
        torch.testing.assert_close([jac1, out], [impl_jac, impl_out])

        # Check class-version
        jac_c, out_c = _to_functional_fmt(getattr(c, f"j{op_name}")(*c_inputs))
        torch.testing.assert_close([jac1, out], [jac_c, out_c])
