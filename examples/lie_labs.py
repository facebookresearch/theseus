import torch

import theseus.labs.lie as lie
import theseus.labs.lie.functional as lieF

batch_size = 5

# ### Lie Tensor creation functions
g1 = lie.SE3.rand(batch_size, requires_grad=True)
print(f"Created SE3 tensor with shape {g1.shape}")
g2 = g1.clone()

# Can create from a tensor as long as it's consistent with the desired ltype
g3_data = lieF.so3.rand(5)
g3 = lie.from_tensor(g3_data, lie.SO3)

try:
    x = lie.from_tensor(torch.zeros(1, 3, 3), lie.SO3)
except ValueError as e:
    print(f"ERROR: {e}")

g4 = lie.as_lietensor(g3_data, lie.SO3)
g5 = lie.cast(g3_data, lie.SO3)  # alias for as_lietensor

# ### Lie operations
v = torch.randn(batch_size, 6)

# Exponential and logarithmic map
out1 = lie.SE3.exp(v)  # also lie.exp(v, g1.ltype)
print(f"Exp map returns a {type(out1)}.")
out2 = g1.log()  # also lie.log(g1)
print(f"Log map returns a {type(out2)}.")

# Inverse
out1 = g1.inv()  # also lie.inv(g1)

# Compose
# also lie.compose(g1, g2)
out1 = g1.compose(g2)  # type: ignore

# Differentiable jacobians
jacs, out = g1.jcompose(g2)  # type: ignore
print("Jacobians output is a 2-tuple.")
print("    First element is a list of jacobians, one per group argument.")
print(f"    For compose this means length {len(jacs)}.")
print("    The second element of the tuple is the result of the operation itself.")
print(f"    Which for compose is a {type(out).__name__}.")

# Other options:
#   * adj(), hat(), vee(), retract(), local(),
#   * Jacobians: jlog(), jinv(), jexp()

# ### Overriden operators
# Compose
out2 = g1 * g2
torch.testing.assert_close(out1, out2, check_dtype=True)

# Transfrom from (from local to world coordinate frame)
p = torch.randn(batch_size, 3)
pt1 = g1.transform_from(p)
pt2 = g1 @ p
torch.testing.assert_close(pt1, pt2)

# For convenience, we provide a context to drop all ltype checks, and operate
# on raw tensor data. However, keep in mind that this is prone to error.
# Here is one example of how this works.
with lie.as_euclidean():
    gg1 = torch.sin(g1)
# The above is the same as this next call, but the context might be more convenient
# if one is doing similar hacky stuff on several group objects.
gg2 = torch.sin(g1._t)
torch.testing.assert_close(gg1, gg2)
print("Success: We just did some ops that make no sense for SE3 tensors.")

# ### Lie tensors can also be used as leaf tensors for torch optimizers
g1 = lie.SE3.rand(1, requires_grad=True)
g2 = lie.SE3.rand(1)

opt = torch.optim.Adam([g1], lr=0.1)

for i in range(10):
    opt.zero_grad()
    d = g1.local(g2)
    loss = torch.sum(d**2)
    loss.backward()
    opt.step()
    print(f"Iter {i}. Loss: {loss.item(): .3f}")
