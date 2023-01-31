from typing import cast

import torch

import theseus.labs.lie as lie

g = lie.rand(5, lie.SE3)
print("LieTensor is tensor", isinstance(g, torch.Tensor), "\n")

# Can retract standard torch tensors
x = torch.rand(5, 6)
z = g.retract(x)

# However, for safety, one cannot use overriden + with torch tensors
try:
    y = g + x
except RuntimeError as e:
    print(e, "\n")


# But one can if one casts them to tangent vectors
w = g + lie.cast(x, ltype=lie.tgt)
torch.testing.assert_close(w._t, z._t)

# One could also do like this, but maybe this is too verbose
y = g + lie.TangentTensor(x, None)
torch.testing.assert_close(z._t, y._t)

# instances of TangentTensor support arbitrary tensor functions. For example
tt = lie.cast(torch.randn(10, 6), ltype=lie.tgt)
out = torch.nn.functional.linear(tt, torch.randn(2, 6), torch.zeros(2))
out = cast(lie.TangentTensor, torch.sigmoid(out))  # cast just for typechecking
print(
    "Any sequence of ops on a tangent tensors results in a",
    type(out).__name__,
    out.ltype,
    "\n",
)
# Or combinations of TangentTensor and standard tensors, for example
a = lie.cast(x, ltype=lie.tgt)
assert torch.allclose(a, x)

# Some torch functions are supported for LieTensor
wz: lie.TangentTensor = torch.cat([w, z])  # type: ignore
assert torch.allclose(wz._t, torch.cat([w._t, z._t]))

# If so all elements must be LieTensors...
try:
    wz = torch.cat([w, z._t])  # type: ignore
except TypeError as e:
    print("\n", e)

# ... of the same ltype
try:
    u = lie.rand(1, lie.SO3)
    wz = torch.cat([w, u])  # type: ignore
except ValueError as e:
    print("\n", e)


# Unsupported operations throw a NotImplementedError
try:
    assert torch.allclose(w, z)
except NotImplementedError as e:
    print(e)

# But you can add lie.as_euclidean() and then everything is valid
with lie.as_euclidean():
    assert torch.allclose(w, z)
    mm = g.matmul(torch.randn(4, 7))
    print(
        "\nWhen calling with lie.as_euclidean(), "
        "the result is not a LieTensor anymore",
        type(mm),
        mm.shape,
    )
