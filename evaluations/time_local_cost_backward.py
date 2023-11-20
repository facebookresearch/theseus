import argparse

import numpy as np
import theseus as th
import torch
import tqdm
import torchlie.functional as lieF
from torchlie.functional.lie_group import LieGroupFns
from torchlie.global_params import set_global_params
from theseus.utils import Timer


def run(
    group_type: str,
    dev: str,
    batch_size: int,
    rng: torch.Generator,
    verbose: bool,
    timer: Timer,
    timer_label: str,
):
    theseus_cls = getattr(th, group_type)
    lieF_cls: LieGroupFns = getattr(lieF, group_type)
    p = torch.nn.Parameter(lieF_cls.rand(batch_size, device=dev, generator=rng))
    adam = torch.optim.Adam([p], lr={"SO3": 0.1, "SE3": 0.01}[group_type])
    a = theseus_cls(name="a")
    b = theseus_cls(
        tensor=lieF_cls.rand(batch_size, device=dev, generator=rng), name="b"
    )
    o = th.Objective()
    o.add(th.Local(a, b, th.ScaleCostWeight(1.0), name="d"))
    layer = th.TheseusLayer(th.LevenbergMarquardt(o, max_iterations=3, step_size=0.1))
    layer.to(dev)
    timer.start(timer_label)
    for i in range(10):
        adam.zero_grad()
        layer.forward(input_tensors={"a": p.clone()}, optimizer_kwargs={"damping": 0.1})
        loss = o.error_metric().sum()
        if verbose:
            print(loss.item())
        loss.backward()
        adam.step()
    timer.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reps", type=int, default=1)
    parser.add_argument("--b", type=int, default=1, help="batch size")
    parser.add_argument("--v", action="store_true", help="verbose")
    parser.add_argument("--g", choices=["SO3", "SE3"], default="SE3", help="group type")
    parser.add_argument("--w", type=int, default=1, help="warmup iters")
    parser.add_argument("--dev", type=str, default="cpu", help="device")
    args = parser.parse_args()

    rng = torch.Generator(device=args.dev)
    rng.manual_seed(0)
    timer = Timer(args.dev)
    print(f"Timing device {timer.device}")

    for p in [True, False]:
        set_global_params({"_allow_passthrough_ops": p})
        set_global_params({"_faster_log_maps": p})
        for i in tqdm.tqdm(range(args.reps + args.w)):
            run(
                args.g,
                args.dev,
                args.b,
                rng,
                args.v,
                timer,
                f"run-{p}" if i > args.w else f"warmup-{p}",
            )
    time_stats = timer.stats()
    results = {}
    for k, v in time_stats.items():
        print([f"{x:.3f}" for x in v])
        results[k] = (np.mean(v), np.std(v) / np.sqrt(len(v)))
        print(k, results[k])
    print(1 - results["run-True"][0] / results["run-False"][0])
    print("-----------------------------")
