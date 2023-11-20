# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import reduce

import torch

from torchlie.global_params import set_global_params


BATCH_SIZES_TO_TEST = [1, 20, (1, 2), (3, 4, 5), tuple()]
TEST_EPS = 5e-7


def get_test_cfg(op_name, dtype, dim, data_shape, module=None):
    atol = TEST_EPS
    # input_type --> tuple[str, param]
    # input_types --> a tuple of type info for a given function
    # all_input_types --> a list of input types, if more than one check isn needed
    all_input_types = []
    if op_name in ["exp", "hat"]:
        all_input_types.append((("tangent", dim),))
        atol = 1e-6 if op_name == "exp" else TEST_EPS
    if op_name == "log":
        all_input_types.append((("group", module),))
        atol = 5e-6 if dtype == torch.float32 else TEST_EPS
    if op_name in ["adjoint", "inverse"]:
        all_input_types.append((("group", module),))
    if op_name == "compose":
        all_input_types.append((("group", module),) * 2)
    if op_name in ["transform", "untransform"]:
        for shape in [(3,)]:
            all_input_types.append((("group", module), ("matrix", shape)))
    if op_name == "lift":
        matrix_shape = (torch.randint(1, 20, ()).item(), dim)
        all_input_types.append((("matrix", matrix_shape),))
    if op_name == "project":
        matrix_shape = (torch.randint(1, 20, ()).item(),) + data_shape
        all_input_types.append((("matrix", matrix_shape),))
    if op_name == "left_act":
        for shape in [
            (3, torch.randint(1, 5, ()).item()),
            (2, 4, 3, torch.randint(1, 5, ()).item()),
        ]:
            all_input_types.append((("group", module), ("matrix", shape)))
    if op_name == "left_project":
        for shape in [
            data_shape,
            ((torch.randint(1, 5, ()).item(),) + data_shape),
        ]:
            all_input_types.append((("group", module), ("matrix", shape)))
    if op_name == "normalize":
        all_input_types.append((("matrix", data_shape),))
        if dtype == torch.float32:
            atol = 2.5e-4
    return all_input_types, atol


# Sample inputs with the desired types.
# Input type is one of:
#   ("tangent", dim)  # torch.rand(*batch_size, dim)
#   ("group", module) # e.g., module.rand(*batch_size)
#   ("quat", dim)     # sampled like tangent but normalized
#   ("matrix", shape) # torch.rand((*batch_size,) + shape)
#
# `batch_size` can be a tuple.
def sample_inputs(input_types, batch_size, dtype, rng):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(batch_size, int):
        batch_size = (batch_size,)

    def _sample(input_type):
        type_str, param = input_type

        def _quat_sample():
            q = torch.rand(*batch_size, param, device=dev, dtype=dtype, generator=rng)
            return q / torch.norm(q, dim=-1, keepdim=True)

        sample_fns = {
            "tangent": lambda: torch.rand(
                *batch_size, param, device=dev, dtype=dtype, generator=rng
            ),
            "group": lambda: param.rand(
                *batch_size, device=dev, generator=rng, dtype=dtype
            ),
            "quat": lambda: _quat_sample(),
            "matrix": lambda: torch.rand(
                (*batch_size,) + param, device=dev, generator=rng, dtype=dtype
            ),
        }
        return sample_fns[type_str]()

    return tuple(_sample(type_str) for type_str in input_types)


# Run some unit tests for a Lie group operator:
# checks:
#   - jacobian of default torch autograd consistent with custom backward implementation
#   - multi-batch output consistent with single-batch output
def run_test_op(op_name, batch_size, dtype, rng, dim, data_shape, module):
    is_multi_batch = not isinstance(batch_size, int)
    bs = len(batch_size) if is_multi_batch else 1
    all_input_types, atol = get_test_cfg(op_name, dtype, dim, data_shape, module=module)
    for input_types in all_input_types:
        inputs = sample_inputs(input_types, batch_size, dtype, rng)
        funcs = (
            tuple(left_project_func(module, x, bs) for x in inputs)
            if op_name == "log"
            else None
        )

        check_lie_group_function(
            module,
            op_name,
            atol,
            inputs,
            funcs=funcs,
            batch_size=batch_size if is_multi_batch else None,
        )


# Checks:
#
#   1) if the jacobian computed by default torch autograd is close to the one
#      provided with custom backward
#   2) if the output of op and jop is consistent with flattening all batch dims
#      to a single dim.
# funcs is a list of callable that modifiies the jacobian. If provided we also
# check that func(jac_autograd) is close to func(jac_custom), for each func in
# the list
def check_lie_group_function(
    module, op_name: str, atol: float, inputs, funcs=None, batch_size=None
):
    op_impl = getattr(module, f"_{op_name}_impl")
    op = getattr(module, f"_{op_name}_autograd_fn")
    jop = getattr(module, f"_j{op_name}_autograd_fn")

    # Check jacobians
    jacs_impl = torch.autograd.functional.jacobian(op_impl, inputs, vectorize=True)
    jacs = torch.autograd.functional.jacobian(op, inputs, vectorize=True)

    if funcs is None:
        for jac_impl, jac in zip(jacs_impl, jacs):
            torch.testing.assert_close(jac_impl, jac, atol=atol, rtol=atol)
    else:
        for jac_impl, jac, func in zip(jacs_impl, jacs, funcs):
            if func is None:
                torch.testing.assert_close(jac_impl, jac, atol=atol, rtol=atol)
            else:
                torch.testing.assert_close(
                    func(jac_impl), func(jac), atol=atol, rtol=atol
                )

    # Check multi-batch consistency
    if batch_size is None:
        return
    lb = len(batch_size)
    flattened_inputs = [x.reshape(-1, *x.shape[lb:]) for x in inputs]
    out = op(*inputs)
    flattened_out = op(*flattened_inputs)
    if jop is None:
        return
    jout = jop(*inputs)[0]
    flattened_jout = jop(*flattened_inputs)[0]
    torch.testing.assert_close(out, flattened_out.reshape(*batch_size, *out.shape[lb:]))
    for j, jf in zip(jout, flattened_jout):
        torch.testing.assert_close(j, jf.reshape(*batch_size, *j.shape[lb:]))


def left_project_func(module, group, batch_dim):
    def func(matrix: torch.Tensor):
        assert matrix.ndim == 2 * batch_dim + 3  # shape should be (*BD, f, *BD, g1, g2)
        g = group.clone()
        # Convert to single-batch-dim sparse gradient format
        batch_size = matrix.shape[:batch_dim]
        if batch_dim > 0:
            d = reduce(lambda x, y: x * y, batch_size)
            matrix = matrix.reshape(d, -1, d, *group.shape[-2:])
            sels = range(matrix.shape[0])
            matrix = matrix[sels, ..., sels, :, :]
            g = group.reshape(d, *group.shape[-2:])
        # Compute projected gradient matrix
        ret = module._left_project_autograd_fn(g, matrix)
        # Revert to multi-batch format if necessary
        if batch_dim > 0:
            ret = ret.reshape(*batch_size, *ret.shape[-2:])
        return ret

    return func


# This function checks that vmap(jacrevc) works for the `group_fns.name`, where
# name can be "exp" or "inv".
# Requires torch >= 2.0
# Compares the output of vmap(jacrev(log(fn(x)))) to jfn(x).
# For "inv" the output of vmap has to be left-projected,
# to make get a Riemannian jacobian.
def check_jacrev_unary(group_fns, dim, batch_size, name):
    assert name in ["exp", "inv"]
    if not hasattr(torch, "vmap"):
        return

    test_fn = getattr(group_fns, name)

    fn_input = (
        torch.randn(batch_size, dim) if name == "exp" else group_fns.rand(batch_size)
    )

    def f(t):
        return group_fns.log(test_fn(t.unsqueeze(0))).squeeze(0)

    j = torch.vmap(torch.func.jacrev(f))(fn_input)
    jac_vmap = group_fns.left_project(fn_input, j) if name != "exp" else j

    jlog, jtest = [], []
    group_fns.log(test_fn(fn_input, jacobians=jtest), jacobians=jlog)
    jac_analytic = jlog[0] @ jtest[0]

    torch.testing.assert_close(jac_vmap, jac_analytic)


# This function checks that vmap(jacrevc) works for the `group_fns.name`, where
# name can be "compose", "transform" and "untransform".
# Requires torch >= 2.0
# Compares the output of vmap(jacrev(log(fn(x)))) to jfn(x).
# For all group inputs, the output of vmap has to be left-projected,
# to make get a Riemannian jacobian.
def check_jacrev_binary(group_fns, batch_size, name):
    assert name in ["compose", "transform", "untransform"]
    if not hasattr(torch, "vmap"):
        return

    test_fn = getattr(group_fns, name)

    fn_inputs = (
        (group_fns.rand(batch_size), torch.randn(batch_size, 3))
        if name in ["transform", "untransform"]
        else (group_fns.rand(batch_size), group_fns.rand(batch_size))
    )

    def f(t1, t2):
        op_out = test_fn(t1.unsqueeze(0), t2.unsqueeze(0))
        if name == "compose":
            op_out = group_fns.log(op_out)
        return op_out.squeeze(0)

    jacs_vmap = []
    for i in range(2):
        j = torch.vmap(torch.func.jacrev(f, i))(fn_inputs[0], fn_inputs[1])
        if fn_inputs[i].ndim == 3:  # group input
            j = group_fns.left_project(fn_inputs[i], j)
        jacs_vmap.append(j)

    jlog, jtest = [], []
    out = test_fn(fn_inputs[0], fn_inputs[1], jacobians=jtest)
    if name == "compose":
        group_fns.log(out, jacobians=jlog)
    for i in range(2):
        jac_analytic = jlog[0] @ jtest[i] if name == "compose" else jtest[i]
        torch.testing.assert_close(jacs_vmap[i], jac_analytic)


def _get_broadcast_size(bs1, bs2):
    m = max(len(bs1), len(bs2))

    def _full_dim(bs):
        return bs if (len(bs) == m) else (1,) * (m - len(bs)) + bs

    bs1_full = _full_dim(bs1)
    bs2_full = _full_dim(bs2)

    return tuple(max(a, b) for a, b in zip(bs1_full, bs2_full))


# flatten to a single batch dimension
def _expand_flat(tensor, broadcast_size, group_size):
    return tensor.clone().expand(broadcast_size + group_size).reshape(-1, *group_size)


def check_binary_op_broadcasting(group_fns, op_name, group_size, bs1, bs2, dtype, rng):
    assert op_name in ["compose", "transform", "untransform"]
    g1 = group_fns.rand(*bs1, generator=rng, dtype=dtype)
    if op_name == "compose":
        t2 = group_fns.rand(*bs2, generator=rng, dtype=dtype)
        t2_size = group_size
    else:
        t2 = torch.randn(*bs2, 3, generator=rng, dtype=dtype)
        t2_size = (3,)

    # The following code does broadcasting manually, then we check that
    # manual broadcast output is the same as the automatic broadcasting
    broadcast_size = _get_broadcast_size(bs1, bs2)
    t1_expand_flat = _expand_flat(g1, broadcast_size, group_size)
    t2_expand_flat = _expand_flat(t2, broadcast_size, t2_size)

    fn = getattr(group_fns, op_name)
    jfn = getattr(group_fns, f"j{op_name}")
    out = fn(g1, t2)
    out_expand_flat = fn(t1_expand_flat, t2_expand_flat)
    torch.testing.assert_close(out, out_expand_flat.reshape(broadcast_size + t2_size))

    jout = jfn(g1, t2)[0]
    jout_expand_flat = jfn(t1_expand_flat, t2_expand_flat)[0]
    for j1, j2 in zip(jout, jout_expand_flat):
        torch.testing.assert_close(j1, j2.reshape(broadcast_size + j1.shape[-2:]))


def check_left_project_broadcasting(
    lie_group_fns, batch_sizes, out_dims, group_size, rng
):
    for bs1 in batch_sizes:
        for bs2 in batch_sizes:
            for out_dim in out_dims:
                g = lie_group_fns.rand(
                    *bs1, generator=rng, dtype=torch.double, requires_grad=True
                )
                t = torch.randn(
                    *bs2,
                    *tuple(range(1, out_dim + 1)),
                    *group_size,
                    generator=rng,
                    dtype=torch.double,
                    requires_grad=True,
                )
                torch.autograd.gradcheck(
                    lie_group_fns.left_project, (g, t, out_dim), raise_exception=True
                )


def check_log_map_passt(lie_group_fns, impl_module):
    set_global_params({"_allow_passthrough_ops": True})
    group = lie_group_fns.rand(
        4, device="cuda:0" if torch.cuda.is_available() else "cpu", requires_grad=True
    )
    jlist = []
    log_map_pt = lie_group_fns.log(group, jacobians=jlist)
    grad_pt = torch.autograd.grad(log_map_pt.sum(), group)
    log_map_ref = impl_module._log_autograd_fn(group)
    jac_ref = impl_module._jlog_impl(group)[0][0]
    grad_ref = torch.autograd.grad(log_map_ref.sum(), group)
    torch.testing.assert_close(log_map_pt, log_map_ref)
    torch.testing.assert_close(jlist[0], jac_ref)
    torch.testing.assert_close(grad_pt, grad_ref)
