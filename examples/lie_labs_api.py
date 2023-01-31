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

# One could also do like this, but maybe this is too verbose, we could just hide this class
y = g + lie.TangentTensor(x)
torch.testing.assert_close(z._t, y._t)

# instances of TangentTensor support arbitrary tensor functions. For example
tt = lie.cast(torch.randn(10, 6), ltype=lie.tgt)
out = torch.nn.functional.linear(tt, torch.randn(2, 6), torch.zeros(2))
out = torch.sigmoid(out)
print(
    "Any sequence of ops on a tangent tensors results in a",
    type(out).__name__,
    out.ltype,
    "\n",
)
# Or combinations of TangentTensor and standard tensors, for example
a = lie.cast(x, ltype=lie.tgt)
torch.testing.assert_close(a, x)

# Some torch functions are supported for LieTensor
wz = torch.cat([w, z])
torch.testing.assert_close(wz._t, torch.cat([w._t, z._t]))

# If so all elements must be LieTensors...
try:
    wz = torch.cat([w, z._t])
except TypeError as e:
    print(e)

# ... of the same ltype
try:
    u = lie.rand(1, lie.SO3)
    wz = torch.cat([w, u])
except ValueError as e:
    print(e)


# Unsupported operations throw a NotImplementedError
try:
    torch.testing.assert_close(w, z)
except NotImplementedError as e:
    print(e)

# But you can add lie.as_eucledian() and then everything is valid
with lie.as_euclidean():
    torch.testing.assert_close(w, z)
