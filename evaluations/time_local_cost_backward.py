import argparse

import theseus as th
import torch
import torchlie.functional as lieF
from torchlie.global_params import set_global_params
from theseus.utils import Timer


def run(
    group_type: str, dev: str, batch_size: int, rng: torch.Generator, verbose: bool
):
    theseus_cls = getattr(th, group_type)
    lieF_cls = getattr(lieF, group_type)
    p = torch.nn.Parameter(lieF_cls.rand(batch_size, device=dev, generator=rng))
    adam = torch.optim.Adam([p], lr={"SO3": 0.1, "SE3": 0.01}[group_type])
    a = theseus_cls(name="a")
    b = theseus_cls(
        tensor=lieF_cls.rand(batch_size, device=dev, generator=rng), name="b"
    )
    o = th.Objective()
    o.add(th.Local(a, b, th.ScaleCostWeight(1.0), name="d"))
    layer = th.TheseusLayer(th.GaussNewton(o, max_iterations=3, step_size=0.1))
    layer.to(dev)
    for i in range(10):
        adam.zero_grad()
        layer.forward(input_tensors={"a": p.clone()})
        loss = o.error_metric().sum()
        if verbose:
            print(loss.item())
        loss.backward()
        adam.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reps", type=int, default=1)
    parser.add_argument("--p", action="store_true")
    parser.add_argument("--dev", type=str, default="cpu")
    parser.add_argument("--v", action="store_true")
    parser.add_argument("--g", choices=["SO3", "SE3"], default="SE3")
    args = parser.parse_args()
    if args.p:
        set_global_params({"_allow_passthrough_ops": True})

    rng = torch.Generator(device=args.dev)
    rng.manual_seed(0)
    batch_sizes = torch.randint(1, 16, (args.reps,), device=args.dev, generator=rng)
    with Timer(args.dev) as timer:
        for i in range(args.reps):
            run(args.g, args.dev, batch_sizes[i].item(), rng, args.v)
    print(timer.elapsed_time)
