from typing import Tuple, Union
import torch


# See Nocedal and Wright, Numerical Optimization, pp. 260 and 261
# https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf
def convert_to_alpha_beta_damping(
    damping: Union[float, torch.Tensor],
    damping_eps: float,
    ellipsoidal_damping: bool,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    damping = torch.as_tensor(damping).to(device=device, dtype=dtype)
    if damping.ndim > 1:
        raise ValueError("Damping must be a float or a 1-D tensor.")
    damping = damping.view(-1)  # this expands floats with ndim = 0
    if batch_size != damping.shape[0]:
        damping = damping.expand(batch_size)
    return (
        (damping, damping_eps * torch.ones_like(damping))
        if ellipsoidal_damping
        else (torch.zeros_like(damping), damping)
    )
