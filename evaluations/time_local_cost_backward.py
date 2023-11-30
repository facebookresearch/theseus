import argparse

import numpy as np
import theseus as th
import torch
import tqdm
import torchlie.functional as lieF
from theseus.global_params import set_global_params as set_th_global_params
from torchlie.functional.lie_group import LieGroupFns
from torchlie.global_params import set_global_params as set_lie_global_params
from theseus.utils import Timer


def run(
    backward: bool,
    group_type: str,
    dev: str,
    batch_size: int,
    rng: torch.Generator,
    verbose_level: int,
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

        def _do():
            layer.forward(
                input_tensors={"a": p.clone()},
                optimizer_kwargs={"damping": 0.1, "verbose": verbose_level > 1},
            )

        if backward:
            adam.zero_grad()
            _do()
            loss = o.error_metric().sum()
            if verbose_level > 0:
                print(loss.item())
            loss.backward()
            adam.step()
        else:
            with torch.no_grad():
                _do()
    timer.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reps", type=int, default=1)
    parser.add_argument("--b", type=int, default=1, help="batch size")
    parser.add_argument("--v", type=int, help="verbosity_level", default=0)
    parser.add_argument("--g", choices=["SO3", "SE3"], default="SE3", help="group type")
    parser.add_argument("--w", type=int, default=1, help="warmup iters")
    parser.add_argument("--dev", type=str, default="cpu", help="device")
    args = parser.parse_args()

    rng = torch.Generator(device=args.dev)
    rng.manual_seed(0)
    timer = Timer(args.dev)
    print(f"Timing device {timer.device}")

    for backward in [True, False]:
        for p in [True, False]:
            label = f"b{backward:1d}-p{p:1d}"
            set_lie_global_params({"_allow_passthrough_ops": p})
            set_lie_global_params({"_faster_log_maps": p})
            set_th_global_params({"fast_approx_local_jacobians": p})
            for i in tqdm.tqdm(range(args.reps + args.w)):
                run(
                    backward,
                    args.g,
                    args.dev,
                    args.b,
                    rng,
                    args.v,
                    timer,
                    f"run-{label}" if i > args.w else f"warmup-{label}",
                )
    time_stats = timer.stats()
    results = {}
    for k, v in time_stats.items():
        results[k] = (np.mean(v), np.std(v) / np.sqrt(len(v)))
        print(k, results[k])
        print([f"{x:.3f}" for x in v])
        print("...............")
    print("With backward pass", 1 - results["run-b1-p1"][0] / results["run-b1-p0"][0])
    print("Only forward pass", 1 - results["run-b0-p1"][0] / results["run-b0-p0"][0])
    print("-----------------------------")
