import torch
from torch.autograd import grad, gradcheck


def check_grad(solve_func, inputs, eps, atol, rtol):
    assert gradcheck(solve_func, inputs, eps=eps, atol=atol)

    A_val, b = inputs[0], inputs[1]
    # Check that the gradient works correctly for floating point data
    out = solve_func(*inputs).sum()
    gA, gb = grad(out, (A_val, b))

    A_float = A_val.float()
    b_float = b.float()
    inputs2 = (A_float, b_float) + inputs[2:]
    print(type(inputs2), len(inputs2))
    out_float = solve_func(*inputs2).sum()
    gA_float, gb_float = grad(out_float, (A_float, b_float))

    torch.testing.assert_close(gA, gA_float.double(), rtol=rtol, atol=atol)
    torch.testing.assert_close(gb, gb_float.double(), rtol=rtol, atol=atol)
