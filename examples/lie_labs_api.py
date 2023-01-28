import torch

import theseus.labs.lie as lie

g = lie.rand(lie.SE3, 5)
x = torch.rand(5, 6)

z = g.retract(x)

try:
    y = g + x
except RuntimeError as e:
    print(e)

y = g + lie.TangentTensor(x)

torch.testing.assert_close(z._t, y._t)

w = g + lie.cast(x, ltype=lie.tgt)
torch.testing.assert_close(w._t, y._t)
