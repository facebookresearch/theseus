# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import os
import pathlib

import random
import time
from typing import Dict, List, Type, cast

import hydra
import numpy as np
import omegaconf
import torch

import theseus as th
import theseus.utils.examples as theg


BACKWARD_MODE = {
    "implicit": th.BackwardMode.IMPLICIT,
    "full": th.BackwardMode.FULL,
    "truncated": th.BackwardMode.TRUNCATED,
}

# Smaller values} result in error
th.SO3.SO3_EPS = 1e-6

# Logger
log = logging.getLogger(__name__)


def print_histogram(
    pg: theg.PoseGraphDataset, var_dict: Dict[str, torch.Tensor], msg: str
):
    log.info(msg)
    histogram = theg.pg_histogram(poses=pg.poses, edges=pg.edges)
    for line in histogram.split("\n"):
        log.info(line)


def get_batch(
    pg: theg.PoseGraphDataset, orig_poses: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    retv = {}
    for pose in pg.poses:
        retv[pose.name] = orig_poses[pose.name]

    return retv


def save_epoch(
    results_path: pathlib.Path,
    epoch: int,
    log_loss_radius: th.Vector,
    theseus_outputs: Dict[str, torch.Tensor],
    info: th.optimizer.OptimizerInfo,
    loss_value: float,
    total_time: float,
):
    def _clone(t_):
        return t_.detach().cpu().clone()

    results = {
        "log_loss_radius": _clone(log_loss_radius.data),
        "theseus_outputs": dict((s, _clone(t)) for s, t in theseus_outputs.items()),
        "err_history": info.err_history,  # type: ignore
        "loss": loss_value,
        "total_time": total_time,
    }
    torch.save(results, results_path / f"results_epoch{epoch}.pt")


def pose_loss(pg: theg.PoseGraphDataset, pose_vars: List[th.LieGroup]) -> torch.Tensor:
    loss: torch.Tensor = torch.zeros(1, dtype=torch.float64)

    for i in range(len(pg.gt_poses)):
        pose_loss = th.local(pose_vars[i], pg.gt_poses[i]).norm(dim=1)
        loss += pose_loss
    return loss


def run(cfg: omegaconf.OmegaConf, results_path: pathlib.Path):
    dtype = torch.float64

    rng = torch.Generator()
    rng.manual_seed(0)

    # create (or load) dataset
    pg, _ = theg.PoseGraphDataset.generate_synthetic_3D(
        num_poses=cfg.num_poses,
        rotation_noise=cfg.rotation_noise,
        translation_noise=cfg.translation_noise,
        loop_closure_ratio=cfg.loop_closure_ratio,
        loop_closure_outlier_ratio=cfg.loop_closure_outlier_ratio,
        generator=rng,
        dtype=dtype,
    )

    # hyper parameters (ie outer loop's parameters)
    log_loss_radius = th.Vector(1, name="log_loss_radius", dtype=dtype)
    robust_loss = th.WelschLoss(log_loss_radius=log_loss_radius, name="welsch_loss")

    # Set up objective
    objective = th.Objective(dtype=torch.float64)

    for edge in pg.edges:
        relative_pose_cost = th.eb.Between(
            pg.poses[edge.i],
            pg.poses[edge.j],
            edge.weight,
            edge.relative_pose,
            loss_function=robust_loss,
        )
        objective.add(relative_pose_cost)

    if cfg.inner_optim.regularize:
        pose_prior_cost = th.eb.VariableDifference(
            var=pg.poses[0],
            cost_weight=th.ScaleCostWeight(
                torch.tensor(cfg.inner_optim.reg_w, dtype=dtype)
            ),
            target=pg.poses[0].copy(new_name=pg.poses[0].name + "__PRIOR"),
        )
        objective.add(pose_prior_cost)

    pose_vars: List[th.LieGroup] = [
        cast(th.LieGroup, objective.optim_vars[pose.name]) for pose in pg.poses
    ]

    if cfg.inner_optim.ratio_known_poses > 0.0:
        pose_prior_weight = th.ScaleCostWeight(100 * torch.ones(1, dtype=dtype))
        for i in range(len(pg.poses)):
            if np.random.rand() > cfg.inner_optim.ratio_known_poses:
                continue
            objective.add(
                th.eb.VariableDifference(
                    pose_vars[i],
                    pose_prior_weight,
                    pg.gt_poses[i],
                    name=f"pose_diff_{i}",
                )
            )

    # Create optimizer
    optimizer_cls: Type[th.NonlinearLeastSquares] = getattr(
        th, cfg.inner_optim.optimizer_cls
    )
    optimizer = optimizer_cls(
        objective,
        max_iterations=cfg.inner_optim.max_iters,
        step_size=cfg.inner_optim.step_size,
    )

    # Set up Theseus layer
    theseus_optim = th.TheseusLayer(optimizer)

    # copy the poses to feed them to each outer iteration
    orig_poses = {pose.name: pose.data.clone() for pose in pg.poses}

    # Outer optimization loop
    loss_radius_tensor = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    model_optimizer = torch.optim.Adam([loss_radius_tensor], lr=cfg.outer_optim.lr)

    num_epochs = cfg.outer_optim.num_epochs

    theseus_inputs = get_batch(pg, orig_poses)
    theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

    with torch.no_grad():
        pose_loss_ref = pose_loss(pg, pose_vars).item()
    log.info(f"POSE LOSS (no learning):  {pose_loss_ref: .3f}")

    print_histogram(pg, theseus_inputs, "Input histogram:")

    for epoch in range(num_epochs):
        log.info(f" ******************* EPOCH {epoch} ******************* ")
        start_time = time.time_ns()
        model_optimizer.zero_grad()
        theseus_inputs["log_loss_radius"] = loss_radius_tensor.unsqueeze(1).clone()

        theseus_outputs, info = theseus_optim.forward(
            input_data=theseus_inputs,
            optimizer_kwargs={
                "verbose": cfg.inner_optim.verbose,
                "track_err_history": cfg.inner_optim.track_err_history,
                "backward_mode": BACKWARD_MODE[cfg.inner_optim.backward_mode],
                "__keep_final_step_size__": cfg.inner_optim.keep_step_size,
            },
        )

        loss = (pose_loss(pg, pose_vars) - pose_loss_ref) / pose_loss_ref
        loss.backward()
        model_optimizer.step()
        loss_value = torch.sum(loss.detach()).item()
        end_time = time.time_ns()

        print_histogram(pg, theseus_outputs, "Output histogram:")
        log.info(
            f"Epoch: {epoch} Loss: {loss_value} "
            f"Kernel Radius: exp({loss_radius_tensor.data.item()})="
            f"{torch.exp(loss_radius_tensor.data).item()}"
        )
        log.info(f"Epoch took {(end_time - start_time) / 1e9: .3f} seconds")

        save_epoch(
            results_path,
            epoch,
            log_loss_radius,
            theseus_outputs,
            info,
            loss_value,
            end_time - start_time,
        )


@hydra.main(config_path="./configs/", config_name="pose_graph")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    results_path = pathlib.Path(os.getcwd())
    run(cfg, results_path)


if __name__ == "__main__":
    main()
